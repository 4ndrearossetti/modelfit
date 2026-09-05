# modelfit

[![crates.io](https://img.shields.io/crates/v/modelfit.svg)](https://crates.io/crates/modelfit)
[![docs.rs](https://docs.rs/modelfit/badge.svg)](https://docs.rs/modelfit)
[![CI](https://github.com/4ndrearossetti/modelfit/actions/workflows/ci.yml/badge.svg)](https://github.com/4ndrearossetti/modelfit/actions)

Answers the question [hwprobe](https://github.com/4ndrearossetti/hwprobe)
measures the ingredients for: **which local AI model should this machine
actually run?**

Detects your hardware (via hwprobe), checks a small curated catalogue of
models against it, recommends the best one (with the reasoning shown, not
just the model choice) and downloads it on request.

```
$ modelfit
machine: discrete GPU | usable for models: 6.0 GiB (+24.8 GiB host for MoE spill)

  Qwen3.8 Flash Next       q95   too big       (needs: 192GB+ unified / serious multi-GPU)
  Qwen3.8 27B              q90   too big       (needs: 24GB VRAM / 32GB+ unified)
  Qwen3.6 35B-A3B          q80   fits (spill)  ~29 tok/s
  Gemma 4 E4B              q60   fits          ~66 tok/s
  Gemma 4 E2B (QAT)        q40   fits          ~125 tok/s
  DeepSeek V4 Flash        q85   too big       (needs: 128GB+ unified / multi-GPU rigs)

recommended: Qwen3.6 35B-A3B (best-quality-spilled)
  get: modelfit get   (unsloth/Qwen3.6-35B-A3B-GGUF — UD-Q4_K_M quant)
  run: llama-server -m <path-to-model.gguf> -c 65536 -b 4096 -ub 4096 --flash-attn auto --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

Usage:

- `modelfit` — recommend for this machine
- `modelfit get [id]` — download the pick (or a named catalogue entry)
  from Hugging Face into `~/models` (`--dir PATH` overrides). Filenames
  are resolved from the repo at download time; interrupted downloads
  resume; `HF_TOKEN` is honoured for gated repos. Prints the run command
  with the real path when done.
- `modelfit show <id>` — one catalogue entry, assessed on this machine
- `--json` — machine-readable output (every assessment carries its
  `launch` object for embedders)
- `--catalog PATH` — your own catalogue instead of the embedded one

## Install

```
cargo install modelfit
```

## How it decides

Two variables:

- **Quality** is a hard-coded score: each catalogue entry carries a
  `quality` integer. Nothing computes it.
- **Speed** is computed per machine: decode is memory-bound, so
  predicted tok/s ≈ bandwidth ÷ bytes-read-per-token. Dense models read
  all their weights every token; MoE models read only the active slice
  (`decode_fraction`). Partially offloaded models read the VRAM-resident
  fraction at GPU speed and stream the rest from host RAM. GPU bandwidth
  is measured per machine when hwprobe reports it (NVML clocks on NVIDIA,
  per-chip table on Apple Silicon), with class constants as fallback;
  host bandwidth is classed by RAM kind when known.

The pick: **highest quality that fits and clears ~20 tok/s** (below that,
a model stops feeling pleasant for interactive use); else the fastest
thing that fits. The reason is always reported, including
`speed-gated-quality`, the answer to "why not the bigger model?".

Predictions are for ordering candidates and gating the floor, not
benchmark truth, so expect them within ~10-25% of measured speed when
hwprobe supplies bandwidth, rougher on class-constant fallback;
conservative on well-tuned setups. Tested memory margins:
max(2 GiB, 9%) of VRAM reserved for the desktop, 20% of unified memory
left to the OS.

## The launch flags

Each catalogue entry carries curated `llama-server` flags (context,
batching, attention, and the model's recommended samplers) authored per
model and, where marked tested, validated on real hardware. The command
is printed for the pick (with the real path after `modelfit get`, with a
placeholder otherwise) and every assessment in the `--json` output
carries its `launch` object for embedders.

Deliberately absent: computed layer placement. llama.cpp's auto-fit
places layers better than precomputed `-ngl`/`-ncmoe` values, so the
templates let it.

## The catalogue

`models.json` — deliberately small and curated, one quant per model (the
size/quality sweet spot, usually dynamic Q4), spanning tiers from
CPU-only 8GB laptops to 192GB+ rigs. Editing it is the intended workflow:
sizes come from the HF repo file listings (sum all shards), quality is
your ordering. The embedded copy makes the binary work offline;
`--catalog` overrides it.
Entries also carry a `launch` template (flags + samplers); `tested: false`
marks flags authored from model defaults rather than validated on hardware.

## Relationship to hwprobe

hwprobe measures (RAM, GPUs, VRAM, unified memory, memory bandwidth);
modelfit suggests (budgets, margins, speed physics, the pick).
If you're building your own recommender, launcher, or installer, you
probably want [hwprobe](https://github.com/4ndrearossetti/hwprobe) as
the base and this repo as a worked example of one opinion layer on top.

## Out of scope

- Running and managing model runtimes (llama.cpp and friends do the
  serving; modelfit stops at the download and the launch command)
- Per-runtime precision (vLLM/MLX memory behaviour differs; predictions
  target the llama.cpp/GGUF world)
- Benchmarking; predictions order candidates, they don't measure

## License

MIT. Recommendation-rule design and memory margins adapted from
hermes-agent (Nous Research, MIT).

