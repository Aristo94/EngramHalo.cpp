#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// QSA indexer fused top-k inputs, see ggml_cuda_try_topk_qsa_fusion in ggml-cuda.cu
struct ggml_cuda_topk_qsa_args {
    const float * score_blk; // [n_blocks, n_tps, n_stream] f32
    const int *   cell_blk;  // [n_kv, n_stream] i32
    const void *  addend;    // [n_kv, n_tps, n_stream] f32/f16, null for none
    bool          add_f16;
    int           n_blocks;
    int           n_tps;
    int           n_stream;
    int           n_kv;
};

void ggml_cuda_op_top_k_qsa(ggml_backend_cuda_context & ctx, const ggml_cuda_topk_qsa_args & args, ggml_tensor * dst);
