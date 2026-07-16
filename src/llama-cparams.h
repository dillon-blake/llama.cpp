#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    // the graph is going to be differentiated, so attention must not route K/V through the
    // KV cache: the cache write is a ggml_set_rows, whose result is a *view* of the cache
    // buffer, which severs the autodiff edge from k_cur/v_cur to the attention output and
    // makes ggml_build_backward_expand abort. set by llama_context::opt_init.
    bool training;

    // Chunked attention (learning-llamas S1-24). 0 = off.
    //
    // Split the QUERY axis of the naive attention into chunks of this many tokens and compute each
    // chunk's softmax independently, concatenating the outputs. The attention matrix is
    // [n_kv, n_q, n_head] -- quadratic in context -- and it is the memory cliff that stops
    // long-context training. Chunking makes only [n_kv, attn_chunk_q, n_head] live at a time.
    //
    // This shrinks the FORWARD residency on its own. It shrinks the BACKWARD's only in company
    // with gradient checkpointing (S1-24 + S1-17), and the reason is worth stating: the backward
    // of a softmax reads its own output, so every chunk's P would otherwise stay live from the
    // forward until the backward consumed it, and chunking would buy exactly nothing. Under
    // ggml_build_backward_expand_checkpointed each chunk's P is a segment-interior node,
    // recomputed immediately ahead of the backward node that reads it and dead again straight
    // after. The two features are multiplicative, not alternatives.
    //
    // Set by llama_context::set_attn_chunk_q. Fixed for the lifetime of a training run: ggml-opt
    // keys its optimizer state by node index, so the graph topology cannot change between steps.
    uint32_t attn_chunk_q;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
