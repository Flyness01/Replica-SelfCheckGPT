# Lessons learned

## 1. A benchmark needs phases, not one headline number

Prompt ingestion and token decoding behaved differently. The first prompt was
processed at 26.104 tokens/second; the next two were processed at 139.774 and
123.265 tokens/second. Decode throughput stayed much closer together, averaging
46.374 tokens/second across the three runs.

That makes “inference speed” too broad to be useful on its own. At minimum, the
benchmark should keep prompt-processing and generation metrics separate.

## 2. Observation is not the same as mechanism

The warm runs were faster during prompt processing, but this baseline does not
isolate why. Model residency, allocation, caching, operating-system state, and
runtime initialization are plausible contributors. Calling the result proof of
one mechanism would require controlled profiling that changes one factor at a
time.

## 3. Repetition is not stochastic sampling

The baseline used MLX-LM's greedy default and produced identical outputs. That
is useful for a reproducible systems baseline, but it is not yet a faithful
SelfCheckGPT-style sampling experiment. A follow-up must use a sampler with a
non-zero temperature and verify that the responses actually vary.

The Python API expects a sampler object rather than `temp=` or `temperature=`
passed directly to `generate()`. The benchmark now constructs one with
`make_sampler()`.

## 4. The prompt exposed a knowledge-boundary failure

The model said that the 2024 Physics prize had not yet been awarded. The Nobel
Prize's official record names John J. Hopfield and Geoffrey Hinton. This makes
the prompt a useful factuality case, but one prompt is not an evaluation set.

## 5. Parallel generation is a trade-off, not a free speedup

Generating samples concurrently may reduce elapsed time, but it can also raise
peak memory, increase coordination overhead, and change throughput. The next
stage needs to compare sequential and batched or concurrent execution under the
same prompt, token limit, sampler, and model state.

## Roadblocks encountered

- Sampling arguments were initially passed to the wrong API layer.
- The first baseline was deterministic, so it measured repeated inference but
  not sample diversity.
- The run recorded a process-level peak-memory number, not a complete memory
  timeline.
- Three runs and one prompt are enough to generate questions, not general
  conclusions.

## Next controlled experiments

1. Compare greedy and stochastic sampling while recording output diversity.
2. Repeat across many prompts and report distributions, not only averages.
3. Separate process startup, model loading, prompt ingestion, and decoding.
4. Compare sequential generation with supported batching or concurrency.
5. Record latency, throughput, peak memory, and factuality together so an
   optimization cannot silently trade away answer quality.
