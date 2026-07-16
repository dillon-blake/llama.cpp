// GLU_BACK's numerics oracle (learning-llamas, S1-28).
//
// MODE_GRAD cannot check this op. mean_abs_asymm divides by (gn + ga), not (|gn| + |ga|), so a
// near-zero gradient element sends the ratio to infinity -- and every GLU gradient is a PRODUCT
// (dx = dy * g * act'(x)), so near-zero elements are ordinary. Measured, the FD "noise" reaches
// MAA 0.80 while a genuinely broken kernel measures 0.18-0.60. They OVERLAP: no tolerance
// separates them. See test_glu::max_maa_err in test-backend-ops.cpp, and ticket S1-37.
//
// So this is the check that actually validates the derivatives: each variant's analytic VJP against
// a FLOAT64 central difference of ggml's OWN scalar forwards (transcribed from ggml-cpu/vec.h --
// not from a paper, because if ggml's gelu uses the tanh approximation then the correct derivative
// is the derivative OF THAT).
//
// It catches what MODE_GRAD cannot: a wrong coefficient. Perturbing GEGLU's tanh argument by 5% is
// invisible to MODE_GRAD and fails here immediately.

// GLU_BACK's analytic derivatives vs a FLOAT64 central difference of ggml's OWN scalar forwards.
// This is the oracle MODE_GRAD cannot be: it is exact to ~1e-7 and it catches a wrong coefficient.
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ggml's scalar forwards, in double, transcribed from ggml/src/ggml-cpu/vec.h.
static const double A = 0.044715, S2PI = 0.79788456080286535587989211986876, S2I = 0.70710678118654752440084436210484;
static double act(int op, double x, double alpha, double limit) {
    switch (op) {
        case GGML_GLU_OP_REGLU:       return x > 0 ? x : 0;
        case GGML_GLU_OP_SWIGLU:      return x/(1.0+exp(-x));
        case GGML_GLU_OP_GEGLU:       return 0.5*x*(1.0+tanh(S2PI*x*(1.0+A*x*x)));
        case GGML_GLU_OP_GEGLU_ERF:   return 0.5*x*(1.0+erf(x*S2I));
        case GGML_GLU_OP_GEGLU_QUICK: return x*(1.0/(1.0+exp(-1.702*x)));
    }
    (void)alpha; (void)limit; return 0;
}
// full forward y = f(x, g)
static double fwd(int op, double x, double g, double alpha, double limit) {
    if (op == GGML_GLU_OP_SWIGLU_OAI) {
        double xc = x < limit ? x : limit;
        double gc = g >  limit ?  limit : (g < -limit ? -limit : g);
        return (xc/(1.0+exp(alpha*(-xc)))) * (gc + 1.0);
    }
    return act(op, x, alpha, limit) * g;
}


// A TRANSPOSED grad must give the same answer as a contiguous one carrying the same values.
//
// ggml's autodiff produces non-contiguous grads routinely (the MUL_MAT backward passes
// ggml_transpose(grad) straight into ggml_out_prod), and no test-backend-ops case ever builds one --
// so this bug is invisible to MODE_GRAD by construction. It was found in the S1-26/S1-27 MoE kernels
// by adversarial review, and GLU_BACK had it too: the same logical grad in two layouts disagreed by
// 1.30. Read src->nb[0]; never index a float*.
static int check_transposed_grad(void) {
    const int NC = 8, NR = 4;
    float out[2][2*8*4];

    for (int pass = 0; pass < 2; ++pass) {
        struct ggml_init_params ip = { 16*1024*1024, NULL, false };
        struct ggml_context * ctx = ggml_init(ip);
        srand(9);

        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NC, NR);
        struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NC, NR);
        for (int i = 0; i < NC*NR; ++i) {
            ((float*)a->data)[i] = 2.f*(rand()/(float)RAND_MAX) - 1.f;
            ((float*)b->data)[i] = 2.f*(rand()/(float)RAND_MAX) - 1.f;
        }
        float g[8*4];
        for (int i = 0; i < NC*NR; ++i) g[i] = 2.f*(rand()/(float)RAND_MAX) - 1.f;

        struct ggml_tensor * dy;
        if (pass == 0) {
            dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NC, NR);
            for (int i = 0; i < NC*NR; ++i) ((float*)dy->data)[i] = g[i];
        } else {
            struct ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NR, NC);
            for (int r = 0; r < NR; ++r) for (int k = 0; k < NC; ++k) ((float*)base->data)[k*NR+r] = g[r*NC+k];
            dy = ggml_transpose(ctx, base);   // ne=[NC,NR], nb[0] = NR*4, NOT 4
        }

        struct ggml_tensor * o = ggml_glu_back(ctx, dy, a, b, GGML_GLU_OP_GEGLU, false, 0.f, 0.f);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_graph_compute_with_ctx(ctx, gf, 2);
        for (int i = 0; i < 2*NC*NR; ++i) out[pass][i] = ((float*)o->data)[i];
        ggml_free(ctx);
    }

    double worst = 0;
    for (int i = 0; i < 2*NC*NR; ++i) worst = fmax(worst, fabs(out[0][i] - out[1][i]));
    printf("  %-12s  transposed grad vs contiguous: max diff %.3e   %s\n",
           "STRIDES", worst, worst == 0.0 ? "identical" : "*** MISREADS A TRANSPOSED GRAD ***");
    return worst != 0.0;
}

int main(void) {
    const int ops[] = {GGML_GLU_OP_REGLU, GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU,
                       GGML_GLU_OP_GEGLU_ERF, GGML_GLU_OP_GEGLU_QUICK, GGML_GLU_OP_SWIGLU_OAI};
    const char * nm[] = {"REGLU","SWIGLU","GEGLU","GEGLU_ERF","GEGLU_QUICK","SWIGLU_OAI"};
    const double alpha = 1.702, limit = 7.0;
    const int N = 64, NR = 5;
    int bad = 0;

    for (int oi = 0; oi < 6; ++oi) {
        const int op = ops[oi];
        struct ggml_init_params ip = { 32*1024*1024, NULL, false };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, NR);  // gate half
        struct ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, NR);  // linear half
        struct ggml_tensor * dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, NR);
        srand(5 + oi);
        for (int i = 0; i < N*NR; ++i) {
            ((float*)a->data)[i]  = 6.0f*(rand()/(float)RAND_MAX) - 3.0f;   // spans kinks & tails
            ((float*)b->data)[i]  = 6.0f*(rand()/(float)RAND_MAX) - 3.0f;
            ((float*)dy->data)[i] = 2.0f*(rand()/(float)RAND_MAX) - 1.0f;
        }

        struct ggml_tensor * out = ggml_glu_back(ctx, dy, a, b, (enum ggml_glu_op) op, false, (float)alpha, (float)limit);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 2);

        double worst_dx = 0, worst_dg = 0, scale_x = 0, scale_g = 0;
        for (int r = 0; r < NR; ++r) {
            const float * dxk = (const float*)out->data + r*2*N;      // half 0 = d_a
            const float * dgk = (const float*)out->data + r*2*N + N;  // half 1 = d_b
            for (int k = 0; k < N; ++k) {
                double x = ((float*)a->data)[r*N+k], g = ((float*)b->data)[r*N+k], d = ((float*)dy->data)[r*N+k];
                // REGLU's kink and SWIGLU_OAI's clamps are non-differentiable points: skip a
                // neighbourhood of them, since the analytic VJP uses a subgradient there by design.
                const double h = 1e-5;
                if (op == GGML_GLU_OP_REGLU && fabs(x) < 1e-3) continue;
                if (op == GGML_GLU_OP_SWIGLU_OAI && (fabs(x-limit) < 1e-3 || fabs(fabs(g)-limit) < 1e-3)) continue;

                double fd_x = d * (fwd(op, x+h, g, alpha, limit) - fwd(op, x-h, g, alpha, limit)) / (2*h);
                double fd_g = d * (fwd(op, x, g+h, alpha, limit) - fwd(op, x, g-h, alpha, limit)) / (2*h);
                // Absolute error, normalized by the tensor's SCALE -- not by the element's own
                // magnitude. silu' has a zero crossing at x ~ -1.278 and gelu' has one too, so a
                // per-element relative error divides by ~0 there and explodes on an exact kernel.
                // That is the same near-zero trap that makes MODE_GRAD's metric unusable; a
                // reference must not repeat it.
                if (fabs(fd_x - dxk[k]) > worst_dx) worst_dx = fabs(fd_x - dxk[k]);
                if (fabs(fd_g - dgk[k]) > worst_dg) worst_dg = fabs(fd_g - dgk[k]);
                if (fabs(fd_x) > scale_x) scale_x = fabs(fd_x);
                if (fabs(fd_g) > scale_g) scale_g = fabs(fd_g);
            }
        }
        double rx = worst_dx/(scale_x > 0 ? scale_x : 1), rg = worst_dg/(scale_g > 0 ? scale_g : 1);
        int ok = rx < 1e-5 && rg < 1e-5;
        if (!ok) bad++;
        printf("  %-12s  worst err / scale:  d_gate %.2e   d_linear %.2e   %s\n",
               nm[oi], rx, rg, ok ? "exact" : "*** WRONG ***");
        ggml_free(ctx);
    }
    bad += check_transposed_grad();

    printf(bad ? "\n%d CHECKS FAILED\n" : "\nall six variants match ggml's own forwards, and strides are honoured\n", bad);
    return bad != 0;
}
