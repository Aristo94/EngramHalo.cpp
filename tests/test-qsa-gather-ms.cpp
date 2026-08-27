#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Multi-sequence exercise of the qwen4exp QSA gather decode path.
//
// llama-perplexity --multiple-choice packs several tasks into one batch: a common
// prefix whose tokens carry several seq_ids at once, followed by per-sequence answer
// tails. split_equal turns the tails into small mixed ubatches (n_tokens <= 16,
// shrinking sequence sets), which is exactly the regime the QSA gather activates in.
// The gather was only ever exercised with single-sequence decode batches before;
// this test pins down the multi-sequence cases:
//
//   A: unified KV cache, coupled common prefix + 4 unequal answer tails
//      (the --multiple-choice shape), small mixed ubatches on the gather path
//   B: non-unified KV cache (n_stream > 1), joint 1-token-per-seq decode batches,
//      including unequal per-stream n_kv
//
// Every scenario is evaluated twice with identical tokens and positions:
//   - reference pass: batched so that every ubatch has > 16 tokens -> dense QSA path
//   - gather pass:    batched so that the flagged tokens sit in <= 16-token ubatches
// Causal attention makes the logits of a token depend only on its history, not on
// how the batch was chunked, so the two passes must agree to FA numerics.
//
// LLAMA_QSA_GATHER=1 is forced (gather active at any n_kv) so that the small
// synthetic model reaches the gather path at all.

struct logits_key {
    llama_seq_id seq;
    llama_pos    pos;

    bool operator<(const logits_key & other) const {
        return seq != other.seq ? seq < other.seq : pos < other.pos;
    }
};

using logits_map = std::map<logits_key, std::vector<float>>;

// decode one batch and collect the logits of all flagged tokens, keyed by (seq_id[0], pos)
static bool decode_collect(llama_context * ctx, llama_batch & batch, int n_vocab, logits_map & out) {
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed (n_tokens=%d)\n", batch.n_tokens);
        return false;
    }

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i]) {
            continue;
        }
        const float * logits = llama_get_logits_ith(ctx, i);
        if (logits == nullptr) {
            fprintf(stderr, "no logits for token %d\n", i);
            return false;
        }
        out[{batch.seq_id[i][0], batch.pos[i]}].assign(logits, logits + n_vocab);
    }

    return true;
}

// normalized mean squared error over all shared keys; both maps must have identical keys
static bool compare_logits(const logits_map & ref, const logits_map & test, double max_nmse, const char * label) {
    if (ref.size() != test.size()) {
        fprintf(stderr, "%s: logit count mismatch: ref %zu vs test %zu\n", label, ref.size(), test.size());
        return false;
    }

    double worst = 0.0;
    logits_key worst_key = {};

    for (const auto & [key, ref_vals] : ref) {
        const auto it = test.find(key);
        if (it == test.end()) {
            fprintf(stderr, "%s: missing logits for seq %d pos %d\n", label, key.seq, key.pos);
            return false;
        }

        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref_vals.size(); ++j) {
            if (!std::isfinite(ref_vals[j]) || !std::isfinite(it->second[j])) {
                fprintf(stderr, "%s: non-finite logit at seq %d pos %d\n", label, key.seq, key.pos);
                return false;
            }
            const double d = ref_vals[j] - it->second[j];
            mse_a_b += d*d;
            mse_a_0 += (double) ref_vals[j]*ref_vals[j];
        }
        const double nmse = mse_a_b/mse_a_0;
        if (nmse > worst) {
            worst     = nmse;
            worst_key = key;
        }
    }

    printf("%s: %zu positions, worst nmse %.3e (seq %d pos %d) -> %s\n",
            label, ref.size(), worst, worst_key.seq, worst_key.pos, worst <= max_nmse ? "OK" : "FAIL");

    return worst <= max_nmse;
}

static llama_token tok_at(llama_seq_id seq, llama_pos pos, int n_vocab) {
    return (llama_token) ((13*(uint32_t) pos + 71*(uint32_t) (seq + 1) + 3) % (uint32_t) n_vocab);
}

// scenario A: unified KV cache, multiple-choice shape
//   - common prefix [0, n_prefix) coupled to all 4 seqs (logits on the last prefix token)
//   - per-seq tails [n_prefix, n_prefix + n_tail) with logits on every token
// gather=false: one decode call, every ubatch > 16 tokens -> dense path
// gather=true:  prefix in one call, tails dribbled in <= 16-token calls with
//               unequal per-seq counts -> small mixed ubatches on the gather path
static bool run_scenario_a(llama_model * model, int n_vocab, bool gather, logits_map & out) {
    constexpr int n_seqs   = 4;
    constexpr int n_prefix = 300; // > 256 so the gather's padded selection still shrinks n_kv
    constexpr int n_tail   = 19;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 1024;
    cparams.n_batch         = 1024;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = n_seqs;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.kv_unified      = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "scenario A: failed to create context\n");
        return false;
    }

    std::vector<llama_seq_id> all_seqs(n_seqs);
    for (int s = 0; s < n_seqs; ++s) {
        all_seqs[s] = s;
    }

    bool ok = true;

    llama_batch batch = llama_batch_init(n_prefix + n_seqs*n_tail, 0, n_seqs);

    if (!gather) {
        // dense reference: everything in one call; the splitter emits the prefix set
        // and the equal-length tails as ubatches of 300 and 76 tokens - never <= 16
        for (llama_pos pos = 0; pos < n_prefix; ++pos) {
            common_batch_add(batch, tok_at(0, pos, n_vocab), pos, all_seqs, pos == n_prefix - 1);
        }
        for (int s = 0; s < n_seqs; ++s) {
            for (int i = 0; i < n_tail; ++i) {
                const llama_pos pos = n_prefix + i;
                common_batch_add(batch, tok_at(s, pos, n_vocab), pos, { (llama_seq_id) s }, true);
            }
        }
        ok = decode_collect(ctx, batch, n_vocab, out);
    } else {
        // gather pass: the prefix is chopped like decode_helper chops at n_batch
        // boundaries - the bulk in one dense call, the last 5 coupled tokens in their
        // own <= 16-token call, so the gather also sees tokens with n_seq_id > 1
        constexpr int n_prefix_cut = n_prefix - 5;

        for (llama_pos pos = 0; pos < n_prefix_cut; ++pos) {
            common_batch_add(batch, tok_at(0, pos, n_vocab), pos, all_seqs, false);
        }
        ok = decode_collect(ctx, batch, n_vocab, out);

        common_batch_clear(batch);
        for (llama_pos pos = n_prefix_cut; pos < n_prefix; ++pos) {
            common_batch_add(batch, tok_at(0, pos, n_vocab), pos, all_seqs, pos == n_prefix - 1);
        }
        ok = ok && decode_collect(ctx, batch, n_vocab, out);

        // then the tails in small joint calls with unequal per-seq counts ->
        // ubatches of 2..16 tokens, seq sets shrinking as sequences run out,
        // all with n_kv >= 512 on the gather path

        int consumed[n_seqs] = {0, 0, 0, 0};

        while (ok) {
            common_batch_clear(batch);

            for (int s = 0; s < n_seqs; ++s) {
                // seq 0 advances fastest, seq 3 slowest - mixed, shrinking seq sets
                const int step = std::min(4 - s, n_tail - consumed[s]);
                for (int i = 0; i < step; ++i) {
                    const llama_pos pos = n_prefix + consumed[s] + i;
                    common_batch_add(batch, tok_at(s, pos, n_vocab), pos, { (llama_seq_id) s }, true);
                }
                consumed[s] += step;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            GGML_ASSERT(batch.n_tokens <= 16 && "scenario A gather pass must stay in gather-sized batches");

            ok = decode_collect(ctx, batch, n_vocab, out);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);

    return ok;
}

// scenario B: non-unified KV cache (one stream per seq), server-like joint decode
//   - two independent sequences with unequal lengths (unequal per-stream n_kv)
//   - logits on the last n_check (+ n_chain for seq 0) tokens of each seq
// gather=false: each seq's tokens in one big call -> dense path
// gather=true:  prefills leave the checked tokens; n_check are decoded in joint
//               1-token-per-seq calls -> 2-token ubatches spanning 2 streams; then
//               seq 0 alone decodes n_chain tokens in one call -> the single-sequence
//               multi-token (spec-decode) shape that stays on the gather path even
//               without LLAMA_QSA_GATHER_MS
static bool run_scenario_b(llama_model * model, int n_vocab, bool gather, logits_map & out) {
    constexpr int n_seqs  = 2;
    constexpr int n_check = 4;
    constexpr int n_chain = 3;
    constexpr int n_len[n_seqs] = {303, 600}; // seq 1 crosses the next n_kv padding step

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 2048; // 1024 cells per stream
    cparams.n_batch         = 1024;
    cparams.n_ubatch        = 1024;
    cparams.n_seq_max       = n_seqs;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.kv_unified      = false;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "scenario B: failed to create context\n");
        return false;
    }

    bool ok = true;

    llama_batch batch = llama_batch_init(*std::max_element(n_len, n_len + n_seqs), 0, 1);

    // seq 0 additionally carries n_chain checked tokens beyond the joint-decode range
    const int n_extra[n_seqs] = {n_check + n_chain, n_check};

    // per-seq prefill; the dense reference keeps the checked tokens inside the big call
    for (int s = 0; s < n_seqs && ok; ++s) {
        common_batch_clear(batch);

        const int n_prefill = gather ? n_len[s] - n_extra[s] : n_len[s];
        for (llama_pos pos = 0; pos < n_prefill; ++pos) {
            const bool flagged = !gather && pos >= n_len[s] - n_extra[s];
            common_batch_add(batch, tok_at(s, pos, n_vocab), pos, { (llama_seq_id) s }, flagged);
        }

        ok = decode_collect(ctx, batch, n_vocab, out);
    }

    if (gather) {
        // joint decode: one token per seq and call -> 2-token ubatches, 2 streams,
        // per-stream n_kv 512 vs 768 after padding
        for (int i = 0; i < n_check && ok; ++i) {
            common_batch_clear(batch);

            for (int s = 0; s < n_seqs; ++s) {
                const llama_pos pos = n_len[s] - n_extra[s] + i;
                common_batch_add(batch, tok_at(s, pos, n_vocab), pos, { (llama_seq_id) s }, true);
            }

            ok = decode_collect(ctx, batch, n_vocab, out);
        }

        // single-sequence multi-token decode (the spec-decode shape): one call,
        // one seq, n_chain tokens -> gather with n_tps > 1 and n_seqs_unq == 1
        if (ok) {
            common_batch_clear(batch);

            for (int i = 0; i < n_chain; ++i) {
                const llama_pos pos = n_len[0] - n_chain + i;
                common_batch_add(batch, tok_at(0, pos, n_vocab), pos, { (llama_seq_id) 0 }, true);
            }

            ok = decode_collect(ctx, batch, n_vocab, out);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);

    return ok;
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx = 1024;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // force the gather on regardless of n_kv and lift the single-sequence restriction:
    // this test exists to validate the multi-sequence envelope behind LLAMA_QSA_GATHER_MS.
    // must precede the first graph build (the gates are read once per process)
#ifdef _WIN32
    _putenv_s("LLAMA_QSA_GATHER",    "1");
    _putenv_s("LLAMA_QSA_GATHER_MS", "1");
#else
    setenv("LLAMA_QSA_GATHER",    "1", 1);
    setenv("LLAMA_QSA_GATHER_MS", "1", 1);
#endif

    llama_backend_init();

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    bool ok = true;

    {
        logits_map ref;
        logits_map res;
        ok = ok && run_scenario_a(model, n_vocab, /*gather =*/ false, ref);
        ok = ok && run_scenario_a(model, n_vocab, /*gather =*/ true,  res);
        ok = ok && compare_logits(ref, res, 1e-4, "scenario A (unified, multiple-choice shape)");
    }

    if (ok) {
        logits_map ref;
        logits_map res;
        ok = ok && run_scenario_b(model, n_vocab, /*gather =*/ false, ref);
        ok = ok && run_scenario_b(model, n_vocab, /*gather =*/ true,  res);
        ok = ok && compare_logits(ref, res, 1e-4, "scenario B (2 streams, joint decode)");
    }

    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", ok ? "ALL OK" : "FAILED");

    return ok ? 0 : 1;
}
