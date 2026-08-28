# Qwen 3.8 Flash-Next on AMD Strix Halo — tuned llama.cpp branch

This branch (`strix-halo-qwen4exp`) sits on the qwen4exp PR
([ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742),
merged upstream on 2026-08-27 as `6c84c7d5`; this branch is rebased onto
master on top of that merge)
plus a small patch series that makes Qwen 3.8 Flash-Next fast on AMD Strix Halo
(Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`, 96 GB unified LPDDR5X).

**Backend scope: ROCm/HIP only.** Everything here is tuned and validated on
ROCm/HIP (`gfx1151`). On Vulkan/RADV this branch is reported to be a net loss:
an unverified third-party report (Reddit, 2026-08-28; Strix Halo, RADV,
UD-IQ4_XS, q8_0 KV — link to follow, not reproduced here) saw short-prompt
prefill drop to roughly half of a stock build and the MTP path collapse to
6–7 t/s decode. The same report says that porting just the MTP graph commits
onto a Vulkan-tuned base keeps stock prefill and gains 26–35% decode with MTP.
Our own Vulkan/RADV cross-check (see BENCHMARKS.md) only covers plain prefill
and decode, not MTP. Until that is reproduced here, treat Vulkan as untested:
start from a stock or Vulkan-tuned build and cherry-pick the MTP graph commits
instead of using this branch as-is.

Measured 2026-08-27 on one machine, q8_0 KV, temperature 0. The model keeps
its 26.8 GiB engram (n-gram) table off the GPU — you choose whether it lives
on **SSD** (mmap load, i.e. the default or `-lm mmap`; ~1 GiB resident, full
262K context) or in **RAM** (`-lm none`, fastest, short contexts only):

| q8_0 KV, temp 0 | stock (IQ3) | tuned IQ3, SSD | tuned IQ3, RAM | tuned IQ4_XS, SSD |
|---|---|---|---|---|
| tg400 code, MTP @ d0 | 24.4 [1] | 35.3 | **39.3** | 31.1 |
| tg300 prose, MTP @ d0 | 22.4 | 25.1 | 25.3 | 23.2 |
| tg300 code, MTP @ d78k | ~10 | 21.3 | — | **24.7** |
| tg300 code, MTP @ d156k | ~6 | **12.1** | — | 11.4 |
| pp4096 @ d0 | 352 | 396 [2] | **496** [2] | **502** [2] |
| pp @ d131k (delta rate) | 91 | 192 | — | — |
| pp average over a 156K prompt | — | 192 | — | **216** |
| resident engram | 26.8 GiB | **~1 GiB** [3] | 26.8 GiB pinned | ~1 GiB [3] |
| max context (single slot) | 262K | **262K** | ~48K | 262K |

Notes: the stock column is plain decode (MTP did not exist there) and is not a
single build — see the build attribution below. `~` values are estimates or
interpolations from the measured depth curve. Depth rows use identical prompts
across the tuned columns.

* **[1]** 24.4 t/s is the plain (no-MTP) decode of the same 400-token code
  prompt in the patched build, i.e. the like-for-like anchor for the 39.3.
  The pre-campaign production config (bf16 KV, `-t 16`, `-ub 512`, 31-token
  generation, measured a day earlier) read 23.5 t/s; the unpatched PR branch
  with the tuned flags read 23.3 t/s.
* **[2]** Three different measurement modes: 396 = server, true SSD-lazy;
  496 = `llama-bench`, `-lm none`; 502 = `llama-bench`, mmap with the engram
  fully page-cached. Not directly comparable — the like-for-like `llama-bench`
  mmap pair is 468 (IQ3) vs 502 (IQ4_XS).
* **[3]** Estimate, not measured on this machine. Expect ~1–1.5 GiB; a DGX
  Spark field report gives 1.4 GiB RSS with the 26.8 GiB table on NVMe.
* **Build attribution for the stock column:** the `@ d0` rows come from
  `b8bdf73bb` with the pre-campaign flags (bf16 KV, `-t 16`, `-ub 512`); the
  depth rows compare against PR branch `243914706` + graph reuse with the
  tuned flags (q8_0 KV, `-t 4`, `-ub 2048`), i.e. the best pre-patch
  configuration.

On quality: the QSA gather rewrite is the only patch that changes decode
numerics, and it measures a wikitext-2 PPL delta of 0.03% against the
dense-mask path in the same build (4.4601 vs 4.4613, 4×32K chunks, identical
text). The remaining patches are kernel-selection and graph-reuse changes; no
end-to-end PPL A/B against the unpatched base was run. With speculative
decoding the target verifies every drafted token, so at temperature 0 the
accepted sequence is the greedy sequence, up to floating-point
nondeterminism from batched verification — not separately measured.

## What the commits do

| Commit | Effect |
|---|---|
| `HIP: wide top-k selection kernel` | The QSA indexer's `top_k` fell back to CPU 12×/token past ~1K context (the "one CPU core at 100%" symptom). Fixes the long-context decode collapse. |
| `CUDA/HIP: skip fully-masked warp slices…` | A backend-generic early exit for fully masked warp slices in the FA vector kernel, which affects every model with masked attention, not just qwen4exp. Also picks the vector kernel on RDNA for the qwen4exp attention shape (hd 256, GQA 2, q8_0 KV). |
| `HIP: chunked GATED_DELTA_NET prefill` | The GDN prefill kernel was token-serial (36 of 48 layers). Chunked kernel, opt-in on RDNA3/RDNA4 via `GGML_HIP_GDN_CHUNK=1` (default-active on NVIDIA Ampere+ and CDNA). It is **not** active in any of the numbers published here. |
| `mmap: prefetch lazily read rows…` | Batched `posix_madvise` readahead for the SSD-backed engram rows: one hint per page-merged row range instead of one page fault per row. The same commit adds an IQ4_NL `get_rows` path for row lengths that are not a multiple of QK_K — without it the 160-value engram gather could not run on the GPU at all. Decode-graph reuse for the qwen4exp inputs is part of the upstream arch since #27742 and is no longer carried here. |
| `qwen4exp: gather top-k KV rows…` | QSA "sparse" attention actually ran dense with a mask — full KV bandwidth at any depth. Decode now gathers the ~2.3K selected rows (top_k 2048 plus the block tail, padded to the FA granularity of 256). Env-gated: `LLAMA_QSA_GATHER=0` disables it, an integer sets the activation threshold in `n_kv` (default 16384, i.e. on from 16K context). Multi-sequence ubatches (`--parallel > 1`, eval tools) fall back to the masked path by default; `LLAMA_QSA_GATHER_MS=1` opts them into the gather for validation runs, and `LLAMA_QSA_GATHER_TRACE=1` logs every gather graph build with its shape. Graph-level correctness of the multi-sequence shapes is pinned by `tests/test-qsa-gather-ms.cpp` on the CPU backend; it is not validated on HIP yet, which is what the default gate is for. |
| `qwen4exp: MTP draft head…` | Multi-token prediction with the draft weights from the official checkpoint. Reference design: [#27739](https://github.com/ggml-org/llama.cpp/pull/27739). |
| `convert: export the qwen4exp MTP block` | Lets you build the MTP sidecar GGUF yourself (see below). |

## Quick start (container, kyuz0-style)

```bash
cd docs/strix-halo
podman build -f Dockerfile.rocm-7.14 -t engramhalo .
toolbox create engramhalo --image localhost/engramhalo -- \
  --device /dev/dri --device /dev/kfd --group-add video --group-add render \
  --security-opt seccomp=unconfined
```

Host prerequisites (96 GB variant): kernel args
`amd_iommu=off amdgpu.gttsize=94208 ttm.pages_limit=24117248` (≈92 GiB GTT)
and the TuneD `accelerator-performance` profile. See
[kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
for the full host setup story.

## Recommended server configs

**A — interactive (single slot, ≤48K context, fastest):**

```bash
toolbox run --container engramhalo \
  env ROCBLAS_USE_HIPBLASLT=1 \
  llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -lm none -c 32768 -b 8192 -ub 2048 -t 4 --parallel 1 --jinja --no-webui \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75
```

No `LLAMA_QSA_GATHER` needed: the gather is on by default from 16K context.
The variable is a threshold in `n_kv`, not an on/off switch — `LLAMA_QSA_GATHER=1`
would force it on from ~2.3K rows upward, a working point none of the numbers
here cover; `0` turns it off.

**B — long context (up to 262144, engram on SSD):**

```bash
toolbox run --container engramhalo \
  env ROCBLAS_USE_HIPBLASLT=1 \
  llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -lm mmap --tensor-read-lazy on -c 262144 -b 8192 -ub 2048 -t 4 \
  --parallel 1 --jinja --no-webui \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75
  # With the MTP sidecar, prefer -c 163840: MTP is validated up to a 164K
  # slot, a 256K slot + MTP was never run. Without -md, the full -c 262144
  # is measured and fine.
```

The only change against A is the load mode: `-lm mmap --tensor-read-lazy on`
with the big context instead of `-lm none -c 32768`. The 51B engram table then
stays SSD-backed (~1 GiB resident instead of 26.8 GiB). `--tensor-read-lazy`
is already `auto` by default and `auto` covers this tensor (every lazy-marked
tensor above 4 GiB), so `on` only makes the intent explicit — the switch that
actually matters is mmap vs. no mmap. Do **not** use `--no-mmap`: it silently
disables the lazy-read path. `-lm none` (RAM mode) is short-context only:
with large slots the first request deadlocks (reproduced with `-c 143360` and
`-c 163840`; `-c 32768` runs cleanly, nothing between 32K and 98K was tested).
The ~48K boundary in the table above is a memory/practice estimate, not a
measured deadlock threshold.

**C — throughput (multiple slots):**

```bash
toolbox run --container engramhalo \
  env ROCBLAS_USE_HIPBLASLT=1 \
  llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -lm mmap --tensor-read-lazy on -c 131072 -b 8192 -ub 2048 -t 4 \
  --parallel 4 --jinja --no-webui
  # -c is the TOTAL context: 131072 / 4 slots = 32K per slot; scale to taste.
  # No MTP sidecar here: multi-slot + speculative decoding is not validated
  # on this arch.
```

Nothing needs to be set for the gather here: multi-slot decode produces
multi-sequence ubatches, and those take the masked path by default (see the
commit table above). Multi-slot serving on gfx1151 does require the #25992
host-buffer workaround (upstream PR #25863, still unmerged);
`Dockerfile.rocm-7.14` applies it. Without it, `--parallel > 1` can return
other requests' responses verbatim. The ≈2.4× aggregate decode at short
contexts was measured pre-patch on the stock build — expect at least the
same, it has not been re-measured here.

Rules of thumb that cost nothing:
* never use bf16 KV cache (the hd-256 FA path re-converts the whole cache
  every call) — `q8_0` is equal speed at half the memory;
* `-ub 2048` needs the current PR head (per-block QSA bias, #27742) — included
  in this branch, but not one of its commits;
* `-t 4` in mmap mode; thread count stops mattering with `-lm none`.

## Building the MTP sidecar

The MTP weights are in the official checkpoint; public GGUFs strip them.
With this branch's converter:

```bash
python convert_hf_to_gguf.py --remote --mtp Qwen/Qwen3.8-Flash-Next \
  --outfile mtp-Qwen3.8-Flash-Next-BF16.gguf --outtype bf16   # ~8 GB download
llama-quantize mtp-Qwen3.8-Flash-Next-BF16.gguf mtp-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0
```

A prebuilt Q8_0 sidecar is at **https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF**. The Q8 sidecar measured *better*
than BF16 (half the draft reads, quant-matched errors → higher acceptance),
and `--spec-draft-p-min 0.75` is what keeps prose from regressing.

## Known limits

* The qwen4exp arch keeps moving upstream; this branch rebases onto master and
  carries the patch series on top of it.
* `llama-perplexity --multiple-choice` crashes on every build of the qwen4exp
  branch (pre-existing upstream bug, patches unrelated).
* PPL with 32K chunks OOMs on the larger quants (logits buffer × 248k vocab) — use 8K chunks.
* n=1 hardware sample; measurement context matters (cold-start runs read low,
  repeated identical requests read high).

## Credits

[#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (Unsloth + llama.cpp
maintainers) · [#27739](https://github.com/ggml-org/llama.cpp/pull/27739)
(Qwen team, MTP reference) ·
[kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
(container pattern, host tuning, FA tile groundwork) ·
[#26592](https://github.com/ggml-org/llama.cpp/pull/26592)/[#26388](https://github.com/ggml-org/llama.cpp/pull/26388)
(hipCUB top-k groundwork) ·
[#27466](https://github.com/ggml-org/llama.cpp/pull/27466)
(radix TOP_K for long rows — the closest prior art to our top-k kernel) ·
[#26001](https://github.com/ggml-org/llama.cpp/pull/26001)/[#20377](https://github.com/ggml-org/llama.cpp/pull/20377)
(chunked GDN prior art) ·
[#27794](https://github.com/ggml-org/llama.cpp/pull/27794) (tensor-read-lazy) ·
[#21458](https://github.com/ggml-org/llama.cpp/pull/21458) (DSA gather op) ·
dzannotti's [MTP GGUF repo](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF) ·
[LaurentZuijdwijk/llama.cpp `vulkan/qwen4exp-rocmfpx`](https://github.com/LaurentZuijdwijk/llama.cpp/tree/vulkan/qwen4exp-rocmfpx)
— first public qwen4exp MTP draft measured on an 8060S (2026-08-26).

Related upstream work (independent; #25917 and #26419 predate this branch,
#27836 was developed in parallel — compare before reusing anything from here):
[#25917](https://github.com/ggml-org/llama.cpp/pull/25917) (sparse KV indices
inside the MMA FA kernel — the kernel-level counterpart to our graph-level QSA
gather) ·
[#26419](https://github.com/ggml-org/llama.cpp/pull/26419) (MMA FA head-dim 256
on AMD RDNA; enables gfx1151 in code, measured on RDNA4 only) ·
[#27836](https://github.com/ggml-org/llama.cpp/pull/27836) (qwen4exp MTP as a
trailing block in the target GGUF — the upstream follow-up path; this branch
uses a standalone sidecar instead) ·
Nathan Wilson's [strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp)
(a Vulkan/RADV-first stack for the same GPU — FA dequant-once (#25494), f16 KV
contiguize, mul_mat_id row-lists, and an independently developed qwen4exp
NextN/MTP loader and draft graph following the deepseek4/deepseek32 pattern;
backend-disjoint from this ROCm/HIP-only branch) ·
Unsloth's `LLAMA_MMAP_RANDOM` batched page prefetch (independent
implementation of the same row-prefetch technique behind the SSD-resident
engram table, reached on the #27742 follow-up branch on 2026-08-27; this
branch drives it from upstream's `TENSOR_READ_LAZY` (#27794) instead) ·
adjacent RDNA 3.5 `ggml-cuda` work by Gaetan Puleo (gated-delta-net tuning,
tiled FA, MMVQ Q8_1 caching) and Nathan Wilson (tile dequant-on-load for
quantized-KV decode) — neighbouring ground to the HIP kernels here, developed
independently.

Model license: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
(distribution permitted with notice; prominent model-name display above 100M
MAU / US$20M monthly revenue; separate license required for Model-as-a-Service
and AI-Work-Assistant businesses — read it).
