// learning-llamas (S1-41): SOFT_MAX_BACK must be correct when its output ALIASES src1.
//
// GGML_OP_SOFT_MAX_BACK is on ggml_op_can_inplace's list, so ggml-gallocr may place the op's
// output on top of one of its sources. It prefers src[0] (the incoming gradient dy), but when
// src[0] is not reusable it falls through to src[1] -- the softmax OUTPUT y. That is not a corner
// case: it is exactly what a Mixtral router backward produces. y feeds only argsort (no gradient)
// and get_rows (which reads the ids, not y), so at backward time SOFT_MAX_BACK is y's sole
// consumer and gallocr reuses y's buffer for the result.
//
// The CPU kernel used to compute dx = scale * y * (dy - <y,dy>) with a `cpy(dx,dy); dx-=<y,dy>;
// dx*=y` vector sequence. That is safe when dx aliases dy (the cpy is then a no-op) but WRONG when
// dx aliases y: the cpy overwrote the whole y buffer before the final `dx*=y` read it, yielding
// ~(dy-<y,dy>)^2. On the router that drove d_logits to nearly zero -- the softmax gate got no
// gradient, and every LoRA upstream of an MoE block trained on a dh that had silently dropped its
// router term. The forward, the loss, and the expert-weight gradients all stayed correct, so
// nothing else caught it. test-backend-ops cannot: it validates CPU against CPU, and both alias
// identically.
//
// This test forces the alias directly and checks the kernel against a float64 reference computed
// from a clean copy of y and dy.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static double frnd(int s) {
    double x = std::sin((double) s * 12.9898) * 43758.5453;
    return (x - std::floor(x)) - 0.5;
}

int main(void) {
    const int64_t n = 4;    // softmax width (a Mixtral has n_expert here)
    const int64_t r = 6;    // rows (tokens)

    std::vector<float> Y(n * r), DY(n * r);
    for (int64_t row = 0; row < r; ++row) {
        // a genuine softmax row, so <y,dy> and the Jacobian are non-degenerate
        double logit[16], mx = -1e30, den = 0;
        for (int64_t i = 0; i < n; ++i) { logit[i] = frnd(1 + row * 10 + i) * 2.0; mx = std::fmax(mx, logit[i]); }
        for (int64_t i = 0; i < n; ++i) { logit[i] = std::exp(logit[i] - mx); den += logit[i]; }
        for (int64_t i = 0; i < n; ++i) { Y[row * n + i] = (float) (logit[i] / den); DY[row * n + i] = (float) frnd(500 + row * 10 + i); }
    }

    struct ggml_init_params ip = { 16 * 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * y  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, r);
    struct ggml_tensor * dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, r);
    std::memcpy(y->data,  Y.data(),  Y.size()  * sizeof(float));
    std::memcpy(dy->data, DY.data(), DY.size() * sizeof(float));

    struct ggml_tensor * dx = ggml_soft_max_ext_back(ctx, dy, y, 1.0f, 0.0f);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, dx);

    // Force what gallocr does for the router: put the output on top of src1 (y).
    dx->data = y->data;

    ggml_graph_compute_with_ctx(ctx, g, 1);
    const float * got = (const float *) dx->data;

    double worst = 0.0;
    for (int64_t row = 0; row < r; ++row) {
        double dot = 0.0;
        for (int64_t i = 0; i < n; ++i) dot += (double) Y[row * n + i] * (double) DY[row * n + i];
        for (int64_t i = 0; i < n; ++i) {
            const double ref = (double) Y[row * n + i] * ((double) DY[row * n + i] - dot);
            worst = std::fmax(worst, std::fabs(ref - (double) got[row * n + i]));
        }
    }

    ggml_free(ctx);

    const bool ok = worst < 1e-4;
    std::printf("soft_max_back output aliased onto src1 (y): worst |kernel - f64 ref| = %.3g  %s\n",
                worst, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
