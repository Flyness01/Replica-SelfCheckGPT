# Replica-SelfCheckGPT

A small systems-research repository for studying the cost of repeated local
generation used by sampling-based hallucination detection methods.

The central question is not only whether a model hallucinates, but what it costs
to test that claim repeatedly: prompt-processing latency, decode throughput,
memory use, and eventually the trade-offs introduced by batching or concurrent
generation.

This is an exploratory benchmark, not a complete reproduction of SelfCheckGPT
and not a claim of a new hallucination-detection method.

## Baseline experiment

The recorded baseline ran three sequential generations in one Python process:

- **Hardware:** Apple Silicon M4 with 16 GB unified memory
- **Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
- **Prompt:** “Who won the Nobel Prize in Physics in 2024?”
- **Generation limit:** 50 tokens
- **Sampling:** greedy (`temperature=0.0`)

| Sample | Prompt tokens/s | Generation tokens/s | Peak memory |
| --- | ---: | ---: | ---: |
| 1 | 26.104 | 44.241 | 1.864 GB |
| 2 | 139.774 | 47.914 | 1.864 GB |
| 3 | 123.265 | 46.966 | 1.864 GB |

The mean warm prompt-processing rate was about **5.04×** the first run, while
generation averaged **46.374 tokens/second**. This is an observation, not proof
of a specific caching mechanism. The baseline does not isolate model residency,
runtime initialization, allocation, or operating-system caching.

All three outputs were identical because the baseline used greedy decoding. The
model incorrectly said the 2024 prize had not yet been awarded; the official
Nobel record identifies [John J. Hopfield and Geoffrey
Hinton](https://www.nobelprize.org/prizes/physics/2024/summary/).

## Run the benchmark

This project uses Python 3.12 and `uv`.

```bash
uv sync
uv run python benchmark.py
```

The default command reproduces the shape of the greedy baseline and writes a
new machine-readable report to `results/latest.json`.

For genuinely varied candidate responses, provide a non-zero temperature:

```bash
uv run python benchmark.py --samples 5 --temperature 0.8 --output results/stochastic.json
```

The MLX-LM Python API accepts sampling behavior through a sampler object. The
script uses `make_sampler()` rather than passing `temp=` or `temperature=`
directly into `generate()`.

Summarize the preserved baseline:

```bash
uv run python analyze_results.py results/baseline_m4.json
```

Run the lightweight analysis tests without loading a model:

```bash
uv run python -m unittest discover -s tests
```

## Repository map

- `benchmark.py` — configurable sequential inference benchmark
- `analyze_results.py` — descriptive summary of saved metrics
- `results/baseline_m4.json` — preserved measurements and generated text
- `notes/lessons-learned.md` — interpretation, limitations, roadblocks, and next experiments
- `tests/test_analysis.py` — tests for metric summarization

## What I learned

1. Prompt ingestion and decoding should be measured separately.
2. A cold/warm difference is evidence of a stateful effect, not automatic proof
   of its cause.
3. Repeating greedy generation does not create independent samples.
4. A useful systems evaluation must track performance and factuality together.
5. Parallel generation may lower latency while increasing memory and
   coordination costs.

The longer reflection and proposed follow-up experiments are in
[`notes/lessons-learned.md`](notes/lessons-learned.md).

## Scope and limitations

- One prompt and three runs cannot support broad performance claims.
- The saved peak-memory value is not a complete memory timeline.
- The recorded run was sequential; it does not benchmark parallel generation.
- The repository preserves the surviving baseline and documented observations,
  not every experiment considered during the research process.

## Research context

SelfCheckGPT introduced a sampling-based approach to detecting hallucinations
without requiring an external knowledge base:

> Manakul, P., Liusie, A., & Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource
> Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP
> 2023.

This repository focuses on the local systems behavior surrounding repeated
generation rather than reproducing the full paper.
