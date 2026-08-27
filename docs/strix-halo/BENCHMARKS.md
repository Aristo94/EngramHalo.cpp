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

* Branch: this repo, `strix-halo-qwen4exp` = ggml-org/llama.cpp
  [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) head
  `af1ffaf37` + the commits of this branch.
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

### Measurement pitfalls (you will hit these)

* **Cold-start reads low:** the first request after server/model load measures
  20–40% slow. Do a warm-up request first.
* **Repeats read high:** sending the identical request again measures ~50%
  fast (prompt cache + the ngram speculator has seen the answer: 52.7 vs 39.3
  t/s in our tests). Always use a fresh prompt for the measured run.
* PPL with 32K chunks OOMs on larger quants (logits buffer × 248,320 vocab ≈
  32 GiB fp32) — use 8K chunks.
* `llama-perplexity --multiple-choice` crashes on every build of the qwen4exp
  branch (pre-existing upstream bug, unrelated to these patches).

## Main result: engram on SSD vs. engram in RAM

The model carries 51B parameters of n-gram ("engram") lookup tables — a single
26.82 GiB tensor that is only ever gathered by input-token id. llama.cpp can
either keep it **SSD-backed** (`-lm mmap --tensor-read-lazy on`, ~1.2 GiB
resident) or **fully resident** (`-lm none`; with the #25992 host-buffer
workaround the gathers then run zero-copy on the iGPU).

ENG build (all patches), IQ3_XXS, q8_0 KV, ub 2048, `ROCBLAS_USE_HIPBLASLT=1`:

| | engram on SSD (`mmap + tensor-read-lazy`) | engram in RAM (`-lm none`) |
|---|---|---|
| prefill @ depth 0 | 395.6 t/s | **495.7 t/s** |
| prefill @ 16K | 381.9 | 376.9 |
| decode @ depth 0 | 22.6 | **24.7** |
| decode @ 16K | 21.0 | 21.5 |
| decode, code, MTP combo | 34.6 | **39.3** |
| decode, prose, MTP combo | 25.2 | 25.3 |
| resident engram footprint | **~1.2 GiB** | 26.8 GiB (pinned) |
| max practical context (single slot, with MTP sidecar) | **262144** (164K validated with MTP) | ~48K |
| load time | seconds (mmap) | minutes (full read + pinning) |
| CPU thread sensitivity | `-t 4` best (CPU-side gathers) | none (t4 ≈ t16; gathers on GPU) |

**Rule of thumb:** RAM mode for interactive/agent work at ≤48K; SSD mode when
you want the full 262K window, the IQ4_XS quant, or memory headroom for other
services. The SSD mode costs ~10% short-context throughput and nothing at depth.

## Depth curves (before / after the patches)

Both curves: same method, same flags (IQ3, q8_0 KV, t4, ub 2048, hipBLASLt),
SSD mode. "Before" = PR branch `243914706` + graph-reuse only. "After" = this
branch.

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

Server, 32K slot, temperature 0, IQ3 target:

| decode t/s | code | prose | acceptance C/P |
|---|---|---|---|
| plain | 23.3 | 22.4 | — |
| BF16 sidecar, n-max 4 | 30.8 | 17.5 | 84% / 38% |
| Q8 sidecar, n-max 4, p-min 0.75 | **35.7** | 23.5 | 92% / 70% |
| Q8 + `draft-mtp,ngram-mod` (recommended) | 34.6 | **25.2** | 80% / 74% |
| ... + `-lm none` (RAM mode), steady state | **39.3** | 25.3 | 83% / — |
| ... @64K depth (SSD mode) | **21.3** (plain: 12.9) | — | 70% |
| ... @156K depth (SSD mode) | **12.1** (plain: ~8.1) | — | 76% |

Lessons: the Q8 sidecar beats BF16 (half the draft reads; quant-matched errors
→ higher acceptance). `--spec-draft-p-min 0.75` is what saves prose. A classic
external 0.8B draft model was measured useless on this architecture (expert
traffic scales with draft depth). `ngram-mod` alone: code 24.7 → 27.8.
Speculation is lossless at temperature 0.

Note: we validated MTP up to a 164K slot. A 256K slot + MTP hung once under
(admittedly self-inflicted) memory pressure and is unverified on a clean box.

## Multi-slot throughput

`llama-batched-bench`, IQ3, q8, npp 2048, ntg 64:

| parallel | pp aggregate | decode aggregate | per stream |
|---|---|---|---|
| 1 | 351.7 | 22.6 | 22.6 |
| 2 | 362.9 | 36.9 | 18.4 |
| 4 | 358.7 | **54.7** | 13.7 |
| 5 | 358.5 | 56.0 (saturated) | 11.2 |

Full-context fit: IQ4_XS @262144/1 slot loads to 71.1 GiB GTT and runs.
IQ3 @4×262144 fits memory-wise (73.8 GiB GTT) but decode collapses (engram
cache squeezed) — use smaller slots for multi-stream.

## Quality

* Wikitext-2 PPL, 16×8K chunks, identical conditions:
  IQ3_XXS **4.1466 ± 0.035** vs IQ4_XS **4.0430 ± 0.034** (IQ4 −2.5%, costs
  8–11% decode).
* Whole patch set (gather vs dense mask, 4×32K chunks, identical text):
  4.4601 vs 4.4613 → **Δ 0.03%**. Top-1 logits identical at 20K depth
  (Δ logprob 0.0009); a dedicated gather-vs-dense regression test pins
  NMSE ≤ 4.4e-14.
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

Server-measured in true SSD-lazy mode (`--tensor-read-lazy on`):

| | IQ3_XXS | IQ4_XS |
|---|---|---|
| pp over a 20.8K prompt | 346.5 | **403.7 (+17%)** |
| decode @ 20K | 20.3 | 19.2 |
| decode, short | 24.4 | 22.5 |
| decode, code, MTP combo | 34.6 | 31.1 |
| decode, prose, MTP combo | 25.2 | 23.2 |

Surprising but consistent: **IQ4_XS prefills faster than IQ3_XXS** (IQ4 dequant
is cheaper in the MMQ path than IQ3 sign-unpacking) while paying ~7% decode.
With its 2.5% better PPL, IQ4_XS is arguably the better default when you can
spare the 11 GiB — it needs SSD mode for large contexts (60.4 GiB GPU part).

Note on comparing tables: llama-bench mmap numbers have the engram fully
cached (no lazy reads), which is why they exceed the lazy-mode server numbers
at depth 0. Both are labeled accordingly.

## RAM mode is a short-context mode (known bug with large slots)

`-lm none` with large slots reproducibly deadlocks on the **first task**:
the server becomes healthy, but the first request sits at
`n_prompt_tokens_processed: 0` with the GPU idle, indefinitely. Reproduced
three times (`-c 98304`, `-c 143360`, `-c 163840`; with and without the MTP
sidecar; once on a freshly rebooted, otherwise idle box). `-c 32768` works
flawlessly and delivers the best short-context numbers. Prime suspect is the
pinned-host compute-buffer path (the #25992 iGPU workaround) with the large
QSA buffers — on the debug list. You lose nothing meanwhile: at 16K depth
both modes are already equal (KV/indexer-bound), so SSD-lazy is both the
only and the lossless option for long contexts.

## Raw data

llama-bench JSONs, server logs, depth-curve CSVs and PPL logs are available on
request (they are per-machine artifacts; open an issue on this fork).
