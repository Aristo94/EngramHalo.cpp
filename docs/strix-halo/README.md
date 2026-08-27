# Qwen 3.8 Flash-Next on AMD Strix Halo — tuned llama.cpp branch

This branch (`strix-halo-qwen4exp`) sits on the qwen4exp PR
([ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742),
merged upstream on 2026-08-27 — a rebase onto master is in progress)
plus a small patch series that makes Qwen 3.8 Flash-Next fast on AMD Strix Halo
(Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`, 96 GB unified LPDDR5X).

**Backend scope: ROCm/HIP only.** Everything here is tuned and validated on
ROCm/HIP (`gfx1151`). On Vulkan/RADV this branch is a net loss: a community
measurement (2026-08-28, Strix Halo, RADV, UD-IQ4_XS, q8_0 KV) saw short-prompt
prefill drop to roughly half of a stock build and the MTP path collapse to
6–7 t/s decode. The same test showed that porting just the MTP graph commits
onto a Vulkan-tuned base keeps stock prefill and gains 26–35% decode with MTP —
so on Vulkan, start from a stock or Vulkan-tuned build and cherry-pick the MTP
graph commits instead of using this branch as-is.

Measured 2026-08-27 on one machine, q8_0 KV, temperature 0. The model keeps
its 26.8 GiB engram (n-gram) table off the GPU — you choose whether it lives
on **SSD** (`-lm mmap --tensor-read-lazy on`, ~1.2 GiB resident, full 262K
context) or in **RAM** (`-lm none`, fastest, short contexts only):

| q8_0 KV, temp 0 | stock (IQ3) | tuned IQ3, SSD | tuned IQ3, RAM | tuned IQ4_XS, SSD |
|---|---|---|---|---|
| tg400 code, MTP @ d0 | 23.5 | 34.6 | **39.3** | 31.1 |
| tg300 prose, MTP @ d0 | 22.4 | 25.2 | 25.3 | 23.2 |
| tg300 code, MTP @ d78k | ~10 | 21.3 | — | **24.7** |
| tg300 code, MTP @ d156k | ~6 | **12.1** | — | 11.4 |
| pp4096 @ d0 | 352 | 396 | **496** | **~502** |
| pp @ d131k | 91 | 192 | — | **216** |
| resident engram | 26.8 GB | **~1.2 GB** | 26.8 GB pinned | ~1.2 GB |
| max context (single slot) | 262K | **262K** | ~48K | 262K |

Notes: stock column = plain decode (MTP did not exist there); `~` values are
estimates/interpolations from the measured depth curve. Depth rows use
identical prompts across the tuned columns.
Quality is untouched: wikitext-2 PPL delta of the whole patch set is 0.03%,
and speculative decoding is lossless at temperature 0.

## What the commits do

| Commit | Effect |
|---|---|
| `HIP: wide top-k selection kernel` | The QSA indexer's `top_k` fell back to CPU 12×/token past ~1K context (the "one CPU core at 100%" symptom). Fixes the long-context decode collapse. |
| `HIP: tune FA path for head-dim 256…` | Kernel selection for the qwen4exp attention shape (hd 256, GQA 2, q8_0 KV). |
| `HIP: chunked GATED_DELTA_NET prefill` | The GDN prefill kernel was token-serial (36 of 48 layers). Opt-in chunked kernel. |
| `qwen4exp: reuse decode graphs…` | No more full rebuild of the ~5k-node graph every token. |
| `qwen4exp: gather top-k KV rows…` | QSA "sparse" attention actually ran dense with a mask — full KV bandwidth at any depth. Decode now gathers the 2048 selected rows. Env-gated: `LLAMA_QSA_GATHER=0` disables it, an integer sets the activation threshold (default on from 16K context). Caveat: multi-sequence ubatches (`--parallel > 1`) currently take the gather path too, and that combination is not separately validated on HIP — set `LLAMA_QSA_GATHER=0` for multi-slot serving until the multi-sequence gate lands. |
| `qwen4exp: MTP draft head…` | Multi-token prediction with the draft weights from the official checkpoint. Reference design: [#27739](https://github.com/ggml-org/llama.cpp/pull/27739). |
| `convert: export the qwen4exp MTP block` | Lets you build the MTP sidecar GGUF yourself (see below). |

## Quick start (container, kyuz0-style)

```bash
cd docs/strix-halo
podman build -f Dockerfile.rocm-7.14 -t llama-strix-qwen4exp .
toolbox create llama-strix-qwen4exp --image localhost/llama-strix-qwen4exp -- \
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
toolbox run --container llama-strix-qwen4exp \
  env ROCBLAS_USE_HIPBLASLT=1 LLAMA_QSA_GATHER=1 \
  llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -lm none -c 32768 -b 8192 -ub 2048 -t 4 --parallel 1 --jinja --no-webui \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75
```

**B — long context (up to 262144):** replace `-lm none -c 32768` with
`-lm mmap --tensor-read-lazy on -c 262144`. The 51B engram table then stays
SSD-backed (~1.2 GiB resident instead of 26.8). Do **not** use `--no-mmap` —
it silently disables the lazy-read path. Note: we validated MTP up to a 164K
slot; 256K slot + MTP is unverified. `-lm none` (RAM mode) is short-context only: with slots >~48K the first request deadlocks (reproducible, on the debug list).

**C — throughput:** config B plus `--parallel 4` with smaller slots
(≈2.4× aggregate decode at short contexts).

Rules of thumb that cost nothing:
* never use bf16 KV cache (the hd-256 FA path re-converts the whole cache
  every call) — `q8_0` is equal speed at half the memory;
* `-ub 2048` needs this branch (per-block QSA bias landed in the PR);
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

* PR #27742 moves quickly; this branch pins a known-good state and will be rebased.
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
[#26001](https://github.com/ggml-org/llama.cpp/pull/26001)/[#20377](https://github.com/ggml-org/llama.cpp/pull/20377)
(chunked GDN prior art) ·
[#27794](https://github.com/ggml-org/llama.cpp/pull/27794) (tensor-read-lazy) ·
[#21458](https://github.com/ggml-org/llama.cpp/pull/21458) (DSA gather op) ·
dzannotti's MTP GGUF repo and the community fork that first ran a qwen4exp
MTP draft on an 8060S.

Model license: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(distribution permitted with notice; MaaS restrictions apply — read it).
