"""Compare repeated full-prefill generation with shared-prefix KV-cache reuse.

This is a latency experiment, not a concurrency benchmark. Both strategies
generate the same number of branches and request the same number of tokens.
With greedy decoding, their outputs must match exactly or the run fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_PROMPT = (
    "Explain who won the Nobel Prize in Physics in 2024 and why their work "
    "was recognized. Be precise and concise."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled shared-prefix KV-cache latency sweep."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--token-levels", nargs="+", type=int, default=[30, 40, 100])
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=Path("results/controlled_cache_sweep.json")
    )
    return parser.parse_args()


def tokenize(tokenizer: Any, prompt: str) -> mx.array:
    tokens = tokenizer.encode(prompt)
    if len(tokens) < 2:
        raise ValueError("The prompt must contain at least two tokens.")
    return mx.array(tokens, dtype=mx.uint32)


def generate_tokens(
    model: Any,
    prompt_tokens: mx.array,
    cache: list[Any],
    max_tokens: int,
    sampler: Any,
) -> list[int]:
    generated = [
        int(token)
        for token, _ in generate_step(
            prompt_tokens,
            model,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=cache,
        )
    ]
    mx.synchronize()
    return generated


def run_sequential(
    model: Any,
    prompt_tokens: mx.array,
    branches: int,
    max_tokens: int,
    sampler: Any,
    seed: int,
) -> dict[str, Any]:
    outputs: list[list[int]] = []
    started = time.perf_counter()
    for branch in range(branches):
        mx.random.seed(seed + branch)
        outputs.append(
            generate_tokens(
                model,
                prompt_tokens,
                make_prompt_cache(model),
                max_tokens,
                sampler,
            )
        )
    elapsed = time.perf_counter() - started
    return {"seconds": elapsed, "outputs": outputs}


def run_shared_prefix(
    model: Any,
    prompt_tokens: mx.array,
    branches: int,
    max_tokens: int,
    sampler: Any,
    seed: int,
) -> dict[str, Any]:
    # Match generate_step's own prefill behavior: cache every prompt token except
    # the final token. Each branch then consumes that final token and generates
    # exactly max_tokens new tokens.
    base_cache = make_prompt_cache(model)
    started = time.perf_counter()
    model(prompt_tokens[:-1][None], cache=base_cache)
    mx.eval([entry.state for entry in base_cache])
    mx.synchronize()
    prefill_seconds = time.perf_counter() - started

    outputs: list[list[int]] = []
    branch_started = time.perf_counter()
    for branch in range(branches):
        mx.random.seed(seed + branch)
        outputs.append(
            generate_tokens(
                model,
                prompt_tokens[-1:],
                copy.deepcopy(base_cache),
                max_tokens,
                sampler,
            )
        )
    branch_seconds = time.perf_counter() - branch_started
    return {
        "seconds": prefill_seconds + branch_seconds,
        "prefill_seconds": prefill_seconds,
        "branch_seconds": branch_seconds,
        "outputs": outputs,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": round(statistics.mean(values), 4),
        "median_seconds": round(statistics.median(values), 4),
        "min_seconds": round(min(values), 4),
        "max_seconds": round(max(values), 4),
    }


def main() -> None:
    args = parse_args()
    if args.branches < 1 or args.repeats < 1:
        raise ValueError("--branches and --repeats must be at least 1.")
    if any(level < 1 for level in args.token_levels):
        raise ValueError("Every token level must be at least 1.")

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)
    prompt_tokens = tokenize(tokenizer, args.prompt)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    results: list[dict[str, Any]] = []

    for max_tokens in args.token_levels:
        print(f"\nToken level: {max_tokens}")
        trials: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic warm-up/order bias.
            order = (
                ("sequential", "shared_prefix")
                if repeat % 2 == 0
                else ("shared_prefix", "sequential")
            )
            measured: dict[str, dict[str, Any]] = {}
            for strategy in order:
                mx.clear_cache()
                trial_seed = args.seed + repeat * args.branches
                if strategy == "sequential":
                    measured[strategy] = run_sequential(
                        model,
                        prompt_tokens,
                        args.branches,
                        max_tokens,
                        sampler,
                        trial_seed,
                    )
                else:
                    measured[strategy] = run_shared_prefix(
                        model,
                        prompt_tokens,
                        args.branches,
                        max_tokens,
                        sampler,
                        trial_seed,
                    )

            outputs_match = (
                measured["sequential"]["outputs"]
                == measured["shared_prefix"]["outputs"]
            )
            if args.temperature == 0.0 and not outputs_match:
                raise RuntimeError(
                    "Greedy outputs differ. The cache comparison is not equivalent."
                )

            seq_seconds = measured["sequential"]["seconds"]
            cached_seconds = measured["shared_prefix"]["seconds"]
            reduction = (seq_seconds - cached_seconds) / seq_seconds * 100
            trial = {
                "repeat": repeat + 1,
                "order": list(order),
                "sequential_seconds": round(seq_seconds, 4),
                "shared_prefix_seconds": round(cached_seconds, 4),
                "shared_prefill_seconds": round(
                    measured["shared_prefix"]["prefill_seconds"], 4
                ),
                "latency_reduction_percent": round(reduction, 2),
                "outputs_match": outputs_match,
                "generated_tokens_per_strategy": args.branches * max_tokens,
            }
            trials.append(trial)
            print(
                f"  repeat {repeat + 1}: sequential={seq_seconds:.3f}s, "
                f"shared-prefix={cached_seconds:.3f}s, reduction={reduction:.1f}%, "
                f"outputs_match={outputs_match}"
            )

        seq_times = [trial["sequential_seconds"] for trial in trials]
        cached_times = [trial["shared_prefix_seconds"] for trial in trials]
        mean_reduction = (
            (statistics.mean(seq_times) - statistics.mean(cached_times))
            / statistics.mean(seq_times)
            * 100
        )
        results.append(
            {
                "max_tokens_per_branch": max_tokens,
                "sequential": summarize(seq_times),
                "shared_prefix": summarize(cached_times),
                "mean_latency_reduction_percent": round(mean_reduction, 2),
                "all_outputs_match": all(t["outputs_match"] for t in trials),
                "trials": trials,
            }
        )

    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_tokens),
        "configuration": {
            "branches": args.branches,
            "repeats": args.repeats,
            "token_levels": args.token_levels,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "experiment": "sequential full-prefill vs sequential shared-prefix reuse",
            "note": "This tests cache reuse, not simultaneous/concurrent execution.",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
