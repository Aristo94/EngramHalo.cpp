# Benchmarks & methodology — Qwen 3.8 Flash-Next on Strix Halo

All numbers measured 2026-08-27 on a single machine. n=1 hardware sample;
measurement context is documented for every series so results are reproducible.

## Hardware & host

| | |
|---|---|
| System | GMKtec EVO-X2, AMD Ryzen AI MAX+ 395 (Strix Halo), 16C/32T Zen 5 |
| GPU | Radeon 8060S iGPU, 40 CU RDNA 3.5, `gfx1151`, wave size 32 |
| RAM | 96 GB unified LPDDR5X-8000 (~256 GB/s theoretical); OS sees 92 GiB after carve-out |
| SSD | Crucial CT1000E100SSD8 NVMe (models on `/home`) |
| OS | Fedora 43, podman/toolbox containers |
| Kernel args | `amd_iommu=off amdgpu.gttsize=94208 ttm.pages_limit=24117248` → ~92 GiB GTT |
| TuneD | `accelerator-performance` (`tuned-ppd` masked so the desktop session cannot reset it) |

GTT is **not** separate VRAM: GPU allocations, CPU processes and the page cache
compete for the same physical 92 GiB pool. This matters for every number below.

## Software

* Branch base: commit `af1ffaf37` of ggml-org/llama.cpp
  [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) — the PR head
  at measurement time on 2026-08-27; the PR was merged later that day as
  `6c84c7d5` — plus the commits of this branch.
* Reference builds measured against: PR-branch commits `b8bdf73bb`
  (build 10678, "stock" starting point) and `243914706` (build 10695).
* ROCm 7.14 (`amdrocm-runtime7.14`, `amdrocm-blas7.14-gfx1151`),
  container recipe: `Dockerfile.rocm-7.14` in this directory.
* Model: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  UD-IQ3_XXS (76.32 GiB; 3 shards) and UD-IQ4_XS (87.24 GiB).
  The engram table (`per_layer_token_embd`) is byte-identical in all quants:
  one IQ4_NL tensor, 26.82 GiB, always CPU-side.
* MTP sidecar: [EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF)
  (Q8_0, 4.1 GB), built with this branch's converter.

## Methodology

* **llama-bench** for flag/quant comparisons:
  `llama-bench -m <model> -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -lm mmap -b 8192 -ub <n> -t <n> -p 4096 -n 128 -d 0,16384 -r 2 -o json`
* **Server timings** for speculation and end-to-end configs: `llama-server` +
  `POST /completion` with `"temperature": 0`, numbers from the `timings` object.
  Code prompt = rewrite-a-function task (400 tokens out), prose prompt =
  descriptive text (300 tokens out, `ignore_eos`).
* **Depth curves** in one pass: one server, one slot, the prompt grows as a
  strict prefix (`cache_prompt: true`), so each stage prefills only its delta.
  pp = delta rate, decode = 64 tokens at that depth. Deepest stage: 236,730
  real tokens.
* **Exclusive box:** all other GPU/RAM services stopped during measurement.
  Never overlap container builds with measurements (we lost a run to the OOM
  killer that way).
* One variable per run, against a named anchor.
* The chunked GDN prefill kernel is opt-in on RDNA3/RDNA4 via
  `GGML_HIP_GDN_CHUNK=1` and was **not** enabled in any run below.

### Measurement pitfalls (you will hit these)

* **Cold-start reads low:** the first request after server/model load measures
  20–55% slow (worst case observed: 17.8 vs 39.3 t/s right after a full
  `-lm none` load). Do a warm-up request first.
* **Repeats read high:** sending the identical request again measures up to
  ~35% fast (prompt cache + the ngram speculator has seen the answer: 52.7 vs
  39.3 t/s in our tests). Always use a fresh prompt for the measured run.
* **Synthetic prompts break MTP decode numbers:** after a random-word filler
  prompt at temperature 0 the model degenerates into repetitive output, the
  ngram speculator hits up to 100% draft acceptance, and decode reads 2–3x
  high (73 t/s observed). Measure MTP decode on real code/prose payloads only;
  prefill (pp) is unaffected by prompt content.
* PPL with 32K chunks OOMs on larger quants (logits buffer × 248,320 vocab ≈
  32 GiB fp32) — use 8K chunks.
* `llama-perplexity --multiple-choice` crashes on every build of the qwen4exp
  branch (pre-existing upstream bug, unrelated to these patches).

## Main result: engram on SSD vs. engram in RAM

The model carries 51B parameters of n-gram ("engram") lookup tables — a single
26.82 GiB tensor that is only ever gathered by input-token id. llama.cpp can
either keep it **SSD-backed** (mmap load plus the lazy tensor read, ~1 GiB
resident) or **fully resident** (`-lm none`). The container applies two loader
and backend workarounds on top of the branch:

* the iGPU host-buffer workaround for llama.cpp issue
  [#25992](https://github.com/ggml-org/llama.cpp/issues/25992), based on the
  still-open PR [#25863](https://github.com/ggml-org/llama.cpp/pull/25863): it
  disables ROCm host-buffer compute on integrated GPUs. That is a correctness
  workaround for the multi-slot response mix-up reported in #25992; it also
  determines where the engram gathers end up being scheduled.
* a per-buffer mmap loader patch: it tracks per shard buffer whether that
  buffer is mmap-backed, drops the blanket `use_mmap` bail-out from the async
  upload path and stops the loader from prefetching every byte of every shard.
  This is what lets the sparse engram tensor stay mmap-backed on the CPU while
  the dense weights are uploaded to the GPU.

> **Note:** both workaround patches were part of the builds these numbers were
> measured on. The published `Dockerfile.rocm-7.14` now ships both and applies
> each one only for as long as it still fits the tree, so a container built
> from it reproduces the measured configuration. An earlier revision of this
> file described the #25992 patch as a "zero-copy" optimization; that
> mislabelled the mechanism (it is a correctness workaround that moves work off
> the host buffer) and has been corrected.

ENG build (all patches), IQ3_XXS, q8_0 KV, ub 2048, `ROCBLAS_USE_HIPBLASLT=1`.
The first four rows are `llama-bench` on both sides — same tool, same metric:

| | engram on SSD (mmap) [a] | engram in RAM (`-lm none`) |
|---|---|---|
| pp4096 @ depth 0 | 468.1 t/s | **491.4 t/s** |
| pp4096 @ 16K | 376.5 | 376.9 [b] |
| tg128 @ depth 0 | 24.6 | **24.7** |
| tg128 @ 16K | 21.5 | 21.4 [b] |
| decode, code, MTP combo (server) | 35.3 | **39.3** |
| decode, prose, MTP combo (server) | 25.1 | 25.3 |
| resident engram footprint | **~1 GiB** [c] | 26.8 GiB (pinned) |
| max practical context (single slot, with MTP sidecar) | **262144** (164K validated with MTP) | ~48K |
| load time | seconds (mmap) | minutes (full read + pinning) |
| CPU thread sensitivity | `-t 4` best (CPU-side gathers) | low (t4 ≈ t16) |

* **[a]** The mmap rows have the engram fully in the page cache. The lazy-read
  path is active there as well; its reads are simply cache hits.
* **[b]** From the `-t 16` run; the other `llama-bench` rows are `-t 4`.
* **[c]** Estimate, not measured on this machine — expect ~1–1.5 GiB. A DGX
  Spark field report gives 1.4 GiB RSS with the 26.8 GiB table on NVMe.

In true SSD-lazy server operation (the table is read from SSD as it is used,
page cache not pre-warmed) the same build delivers 395.6 t/s prefill and
22.6 t/s decode at 4K depth, 381.9 / 21.0 at 16K. Those are depth-curve delta
rates, a different metric from `llama-bench` `pp4096` — do not compare them
against the table above.

**Rule of thumb:** RAM mode for interactive/agent work at ≤48K; SSD mode when
you want the full 262K window, the IQ4_XS quant, or memory headroom for other
services. Like-for-like, SSD mode costs ~5% prefill at depth 0 and nothing on
decode; the true lazy-read server numbers are lower because the table is read
from SSD as it is used.

## Depth curves (before / after the patches)

Both curves: same method, same flags (IQ3, q8_0 KV, t4, ub 2048, hipBLASLt),
SSD mode. "Before" = PR branch `243914706` plus the graph-reuse commit — which
also carries the engram row prefetch and the IQ4_NL `get_rows` path, so those
two are already in the baseline. "After" = this branch.

| depth (tokens) | pp before | pp after | decode before | decode after |
|---|---|---|---|---|
| 4K | 362.2 | 395.6 | 21.82 | 22.57 |
| 16K | 271.0 | 381.9 | 18.62 | 20.96 |
| 32K | 201.7 | 317.2 | 16.06 | 18.89 |
| 64K | 148.5 | 258.5 | 11.00 | 13.52 |
| 96K | 111.6 | 219.7 | 8.65 | 11.26 |
| 131K | 90.9 | 192.3 | 7.02 | 9.53 |
| 160K | — | 173.5 | — | 8.14 |
| 192K | — | 155.4 | — | 7.25 |
| 224K (236.7K real) | — | 137.9 | — | 6.45 |

The remaining depth decay is the QSA indexer itself (it still scores O(ctx/4)
blocks per token) — the next structural target.

## Depth curves on the shipped branch (2026-08-28, single build)

Measured on the exact tree this branch ships (pre-rebase equivalent of the
current HEAD; content-identical patches). Different method than the
before/after table above: `llama-bench -p 2048 -n 64 -d <depth> -r 1`
(marginal cost *at* the depth, not a server average), q8_0 KV, t4, ub 2048,
hipBLASLt, SSD mode, cold page cache per run.

| depth | IQ3 pp2048 / tg64 | IQ4_XS pp2048 / tg64 |
|---|---|---|
| 4K | 470.9 / 22.5 | 478.7 / 20.9 |
| 8K | 435.7 / 21.2 | 441.7 / 19.9 |
| 16K | 383.3 / 20.4 | 382.2 / 19.0 |
| 32K | 315.6 / 17.9 | 318.8 / 17.0 |
| 64K | 248.5 / 13.2 | 240.8 / 12.2 |
| 128K | 174.4 / 9.1 | 169.1 / 8.3 |

Prefill is quant-independent across the whole curve (IQ4 even leads at
shallow depth), and the IQ4 decode tax shrinks with depth (−7% at 4K, −9% at
128K) as attention/indexer cost dominates the weights.

**MTP decode at depth (server, real code payloads, temperature 0, Q8_0
sidecar, `draft-mtp,ngram-mod`, n-max 4, p-min 0.75):**

| context depth | decode | draft acceptance | vs. plain decode |
|---|---|---|---|
| ~0K (32K slot) | 36.4 t/s | 79.6% | +47% (24.7) |
| 77.7K (131K slot) | 21.6 t/s | 73.5% | +63% (13.2 @64K) |
| 156.4K (163840 slot) | 20.8 t/s | 66.0% | >2x (9.1 @128K) |

The MTP win *grows* with depth: each accepted draft token amortizes the
per-step attention/indexer cost. A 156K-token prompt prefills in ~12 minutes
(221.7 t/s average over the whole prompt with the sidecar loaded) and then
decodes at 20+ t/s. Acceptance degrades only mildly with depth.

Prefill with the full MTP stack loaded (server, `cache_prompt=false`, fresh
~4–86K-token prompts): 464 t/s avg at 5.4K, 402 at 21K, 342 at 43K, 272 at
86K — within a few percent of the plain llama-bench curve, i.e. the sidecar
costs prefill essentially nothing.

## Flag matrix (stock build `b8bdf73bb`, the free wins)

Starting config was `bf16` KV, `-t 16`, `-ub 512`:

| change (one variable) | pp4096 @0 | tg128 @0 | pp @16K | tg @16K |
|---|---|---|---|---|
| baseline (bf16 KV, t16, ub512) | 352.5 | 24.6 | 217.0 | 13.4 |
| `-t 4` | — | 24.2 | — | **15.7** |
| KV `f16` (t4) | 317* | 23.9 | 197.6 | **18.5** |
| KV `q8_0` (t4) | 328.4 | 24.5 | 197.3 | **18.1** |
| `ROCBLAS_USE_HIPBLASLT=1` (pp only, r=3) | **342.4** vs 326.1 | — | — | — |
| UD-IQ4_XS instead of IQ3 (q8, t4) | 281* | 21.7 | 167.3 | 16.6 |

\* cold-cache-tainted first runs.

Why bf16 KV was the single biggest mistake: with head-dim 256 the decode takes
the TILE FA kernel, which re-converts the **entire** bf16 K/V cache to f16 on
every call (`need_f16_K/V`). f16 is the conversion-free path; q8_0 is equal
speed at half the memory (KV+indexer @262144: 8.3 → ~4.5 GiB).

`-ub 2048` (needs the per-block QSA bias from the current PR head):
pp 328.7 → 390.6 (ub 1024) → **401.7** (ub 2048) @depth 0;
200.2 → 223.6 → **230.1** @16K.

## Speculation (MTP)

Server, 32K slot, temperature 0, IQ3 target. Rows 1–4 were measured on the
MTP-port image (`af1ffaf37` + the MTP commits only, without the kernel and
gather patches), which is why they isolate the speculation effect; rows 5–7
are the full patch set. In the full build the recommended combo reads 35.3
code / 25.1 prose.

**Provenance note (2026-08-28):** the campaign binaries behind this table
predate the qwen4exp MTP head shipped in this branch — the sidecar ran through
the draft path of that day's PR-branch state. The head as shipped here first
went through an acceptance run on 2026-08-28; its first request crashed (an
output-masking bug in the unmasked nextn readback, fixed inside the MTP
commit), after which it reads code 34.3 t/s at 79.6% acceptance and prose
22.2 t/s at 63.9% (32K slot, Q8_0 sidecar, temperature 0; plain baseline in
the same build: 24.4 t/s). Plain mainline master cannot load the sidecar at
all until an upstream follow-up (#27836 or equivalent) lands.

| decode t/s | code | prose | acceptance C/P |
|---|---|---|---|
| plain (MTP image) | 23.3 | 22.4 | — |
| BF16 sidecar, n-max 4 | 30.8 | 17.5 | 84% / 38% |
| Q8 sidecar, n-max 4, p-min 0.75 | **35.7** | 23.5 | 92% / 70% |
| Q8 + `draft-mtp,ngram-mod` (recommended) | 34.6 | **25.2** | 80% / 74% |
| ... + `-lm none` (RAM mode), full build, steady state | **39.3** | 25.3 | 83% / — |
| ... @78K depth, 77,669 tokens (SSD mode) | **21.3** (plain: 12.9) | — | 70% |
| ... @156K depth (SSD mode) | **12.1** (plain: ~8.1) | — | 76% |

Lessons: the Q8 sidecar beats BF16 (half the draft reads; quant-matched errors
→ higher acceptance). `--spec-draft-p-min 0.75` is what saves prose. A classic
external 0.8B draft model was measured useless on this architecture (expert
traffic scales with draft depth). `ngram-mod` alone: code 24.7 → 27.8
(canreuse build, 32K slot; the plain anchor there was 24.7, not the 23.3 of
the table above).

On losslessness: the target verifies every drafted token, so at temperature 0
the accepted sequence is the greedy sequence, up to floating-point
nondeterminism from batched verification. Not separately measured — there is
no output diff or logit comparison with the draft head enabled.

Note: MTP was validated up to a 163,840-token slot (SSD mode). A 98K slot in
RAM mode with the sidecar did not fit (92/92 GiB, thrash). 256K + MTP was
never run.

## Multi-slot throughput

`llama-batched-bench` on the **stock build `b8bdf73bb`**, IQ3, q8, `-ub 512`,
`-c 16384`, npp 2048, ntg 64 — not re-measured on this branch:

| parallel | pp aggregate | decode aggregate | per stream |
|---|---|---|---|
| 1 | 351.7 | 22.6 | 22.6 |
| 2 | 362.9 | 36.9 | 18.4 |
| 4 | 358.7 | **54.7** | 13.7 |
| 5 | 358.5 | 56.0 (saturated) | 11.2 |

Full-context fit: IQ4_XS @262144/1 slot loads to 71.1 GiB GTT and runs.
IQ3 @4×262144 fits memory-wise (73.8 GiB GTT) but decode collapses (engram
cache squeezed) — use smaller slots for multi-stream.

**Correctness caveat for multi-slot on gfx1151:** `-np > 1` with `--kv-unified`
is exactly the configuration of llama.cpp issue
[#25992](https://github.com/ggml-org/llama.cpp/issues/25992) — requests can
receive other requests' responses verbatim. The fix (PR
[#25863](https://github.com/ggml-org/llama.cpp/pull/25863)) is still unmerged
upstream and is **not** in this tree; `Dockerfile.rocm-7.14` applies it at
build time. If you build this branch outside that container, do not run
multi-slot. The QSA gather needs no attention here: multi-sequence ubatches
take the masked path by default, so multi-slot serving never enters the
gather unless `LLAMA_QSA_GATHER_MS=1` is set.

## Quality

* Wikitext-2 PPL, 16×8K chunks, identical conditions:
  IQ3_XXS **4.1466 ± 0.035** vs IQ4_XS **4.0430 ± 0.034** (IQ4 −2.5%, costs
  8–11% decode on the stock build; ~7% with these patches, see the IQ3/IQ4
  matrix below).
* **QSA gather vs. dense mask**, same build, env-gate A/B (4×32K chunks,
  identical text): 4.4601 vs 4.4613 → **Δ 0.03%**. The gather is the only
  patch that touches decode numerics; the rest are kernel-selection and
  graph-reuse changes, and no end-to-end PPL A/B of the full patch set against
  the unpatched base was run. Top-1 logits identical at 20K depth
  (Δ logprob 0.0009); the checked-in gather-vs-dense regression test
  `tests/test-qsa-gather-ms.cpp` runs under ctest on the CPU backend and pins
  NMSE ≤ 4.4e-14 across the multi-sequence shapes.
* Backend cross-check: Vulkan/RADV on the same commit measured pp 206/151 and
  decode 24.8/17.9 (@0/@16K) — ROCm with these patches wins everywhere.
  (Vulkan has since gained lightning-indexer kernels upstream; expect that gap
  to move.)

## IQ3_XXS vs IQ4_XS (ENG build)

llama-bench pair, mmap with the engram fully page-cached (q8_0 KV, t4, ub 2048):

| | IQ3_XXS | IQ4_XS |
|---|---|---|
| pp4096 @ depth 0 | 468.1 | **501.6** |
| pp4096 @ 16K | 376.5 | 379.3 |
| tg128 @ depth 0 | 24.6 | 22.8 |
| tg128 @ 16K | 21.5 | 20.1 |

Server-measured with a cold-ish page cache, i.e. the engram really is read
from SSD as it is used:

| | IQ3_XXS | IQ4_XS |
|---|---|---|
| pp over a 20.8K prompt | 346.5 | **403.7 (+17%)** |
| decode @ 20K | 20.3 | 19.2 |
| decode, short | 24.4 | 22.5 |
| decode, code, MTP combo | 35.3 | 31.1 |
| decode, prose, MTP combo | 25.1 | 23.2 |
| decode @ 78K depth, MTP (identical prompt) | 21.3 | **24.7** |
| decode @ 156K depth, MTP (identical prompt) | **12.1** | 11.4 |
| prefill, average over the 156K prompt | 192.3 | **215.6** |

Surprising but consistent: **IQ4_XS prefills faster than IQ3_XXS** (IQ4 dequant
is cheaper in the MMQ path than IQ3 sign-unpacking) while paying ~7% decode.
With its 2.5% better PPL, IQ4_XS is arguably the better default when you can
spare the 11 GiB — it needs SSD mode for large contexts (60.4 GiB GPU part).

Note on comparing tables: the `llama-bench` mmap numbers have the engram fully
in the page cache, which is why they exceed the lazy-mode server numbers at
depth 0. The lazy-read path is active in both — `--tensor-read-lazy` defaults
to `auto`, and `auto` already covers every lazy-marked tensor above 4 GiB, so
passing `on` changes nothing for this tensor. The difference between the two
tables is a warm vs. a cold page cache, not lazy on vs. off.

## RAM mode is a short-context mode (known bug with large slots)

`-lm none` with large slots deadlocks on the **first task**: the server becomes
healthy, but the first request sits at `n_prompt_tokens_processed: 0` with the
GPU idle, indefinitely. Reproduced with `-c 143360` and `-c 163840`, with and
without the MTP sidecar. A `-c 98304` slot in RAM mode failed earlier for a
different reason — it did not fit (92/92 GiB, thrash). `-c 32768` works
flawlessly and delivers the best short-context numbers; nothing between 32K
and 98K was tested, so the ~48K boundary quoted elsewhere is a memory/practice
estimate rather than a measured deadlock threshold. Prime suspect is the
pinned-host compute-buffer path (the #25992 iGPU workaround) with the large
QSA buffers — on the debug list. You lose nothing meanwhile: at 16K depth
both modes are already equal (KV/indexer-bound), so SSD-lazy is both the
only and the lossless option for long contexts.

## Raw data

llama-bench JSONs, server logs, depth-curve CSVs and PPL logs are per-machine
artifacts and are not committed. Open an issue on this fork if you want them,
or ask via the discussions of the
[HF sidecar repo](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF/discussions).
