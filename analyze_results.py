"""Summarize a saved benchmark without making causal claims."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    samples = report.get("samples", [])
    if not samples:
        raise ValueError("The report contains no samples.")

    prompt_speeds = [sample["prompt_tps"] for sample in samples]
    generation_speeds = [sample["generation_tps"] for sample in samples]
    outputs = [sample["text"] for sample in samples]
    summary = {
        "sample_count": len(samples),
        "mean_generation_tps": round(statistics.fmean(generation_speeds), 3),
        "identical_outputs": len(set(outputs)) == 1,
        "peak_memory_gb": max(sample["peak_memory_gb"] for sample in samples),
    }

    if len(prompt_speeds) > 1 and prompt_speeds[0] > 0:
        warm_speeds = prompt_speeds[1:]
        summary["cold_prompt_tps"] = prompt_speeds[0]
        summary["mean_warm_prompt_tps"] = round(statistics.fmean(warm_speeds), 3)
        summary["warm_to_cold_prompt_ratio"] = round(
            statistics.fmean(warm_speeds) / prompt_speeds[0], 3
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a benchmark JSON file.")
    parser.add_argument("report", type=Path, nargs="?", default=Path("results/baseline_m4.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(json.dumps(summarize(report), indent=2))


if __name__ == "__main__":
    main()
