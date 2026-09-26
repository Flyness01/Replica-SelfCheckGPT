"""Run a repeatable sequential inference baseline with MLX-LM.

This script measures repeated generations in one process. It deliberately does
not call the run "parallel" or attribute warm-run improvements to a particular
cache mechanism; those questions require separate controlled experiments.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_PROMPT = "Who won the Nobel Prize in Physics in 2024?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure repeated local MLX-LM inference runs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 reproduces the greedy baseline; use a value above 0 for stochastic samples.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/latest.json"))
    return parser.parse_args()


def generate_sample(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Generate one response and retain the final metrics reported by MLX-LM."""
    sampler = make_sampler(temp=temperature, top_p=top_p)
    text_parts: list[str] = []
    final_response = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        text_parts.append(response.text)
        final_response = response

    if final_response is None:
        raise RuntimeError("MLX-LM returned no generation response.")

    return {
        "text": "".join(text_parts),
        "prompt_tokens": final_response.prompt_tokens,
        "prompt_tps": round(final_response.prompt_tps, 3),
        "generation_tokens": final_response.generation_tokens,
        "generation_tps": round(final_response.generation_tps, 3),
        "peak_memory_gb": round(final_response.peak_memory, 3),
        "finish_reason": final_response.finish_reason,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be greater than 0 and at most 1")

    model, tokenizer = load(args.model)
    samples = []
    for index in range(args.samples):
        print(f"Generating sample {index + 1}/{args.samples}...")
        result = generate_sample(
            model,
            tokenizer,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_p,
        )
        result["sample"] = index + 1
        samples.append(result)

    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "model": args.model,
        "prompt": args.prompt,
        "configuration": {
            "samples": args.samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "execution": "sequential, repeated in one process",
        },
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved benchmark report to {args.output}")


if __name__ == "__main__":
    main()
