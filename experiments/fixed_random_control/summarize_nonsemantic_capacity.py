#!/usr/bin/env python
"""Aggregate completed fixed-random control evaluations without changing them.

The script accepts the evaluation JSON files produced by
``evaluate_nonsemantic_capacity.py`` and writes a compact, auditable summary.
It reports population and sample standard deviations; the paper uses the
sample standard deviation for five-seed mean+-std tables.  The script never
selects a target-domain threshold.

Paired comparisons use the visual-only (None) and canonical CMCE source-only
per-seed values reported in the paper's transfer tables, listed below.

Example::

    python experiments/fixed_random_control/summarize_nonsemantic_capacity.py \
        --eval-json results/fixed_random_control/seed*/evaluation_inria_mass.json \
        --output results/fixed_random_control/five_seed_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


EXPECTED_SEEDS = [0, 42, 123, 456, 666]
BASELINE_NONE = {
    42: {"inria": 41.94, "massachusetts": 28.76},
    0: {"inria": 50.20, "massachusetts": 24.31},
    123: {"inria": 50.66, "massachusetts": 24.08},
    456: {"inria": 46.04, "massachusetts": 20.58},
    666: {"inria": 44.50, "massachusetts": 26.79},
}

# Source-only transfer IoU of the released CMCE checkpoints (weights/cmce_whu_seed*.pth).
CMCE_INRIA = {
    0: 0.5034198843457413,
    42: 0.5405460259095294,
    123: 0.5217835846795945,
    456: 0.5161083382050952,
    666: 0.4716423739727793,
}
CMCE_MASSACHUSETTS = {
    0: 0.28512736012780965,
    42: 0.2974910877671512,
    123: 0.32590686523093393,
    456: 0.29837190382266493,
    666: 0.2823015901074485,
}


def mean_std(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "std_sample": float(array.std(ddof=1)) if len(array) >= 2 else 0.0,
    }


def paired_summary(control: Dict[int, float], baseline: Dict[int, float]) -> Dict[str, Any]:
    seeds = sorted(set(control) & set(baseline))
    diffs = np.asarray([control[s] - baseline[s] for s in seeds], dtype=float)
    result: Dict[str, Any] = {
        "seeds": seeds,
        "differences_control_minus_baseline": diffs.tolist(),
        **mean_std(diffs),
    }
    if len(diffs) >= 2 and float(diffs.std(ddof=1)) > 0:
        t_value = float(diffs.mean() / (diffs.std(ddof=1) / math.sqrt(len(diffs))))
        result["paired_t"] = t_value
        result["df"] = len(diffs) - 1
        try:
            from scipy.stats import t as student_t

            result["paired_p_two_sided"] = float(2.0 * student_t.sf(abs(t_value), len(diffs) - 1))
            critical = float(student_t.ppf(0.975, len(diffs) - 1))
            half_width = critical * float(diffs.std(ddof=1)) / math.sqrt(len(diffs))
            result["mean_difference_ci95"] = [
                float(diffs.mean() - half_width),
                float(diffs.mean() + half_width),
            ]
        except Exception:
            # The t statistic remains useful without scipy; do not invent a p-value.
            result["paired_p_two_sided"] = None
        positives = int((diffs > 0).sum())
        negatives = int((diffs < 0).sum())
        non_ties = positives + negatives
        if non_ties:
            tail = sum(math.comb(non_ties, k) for k in range(max(positives, negatives), non_ties + 1)) / float(2 ** non_ties)
            result["sign_test_two_sided"] = float(min(1.0, 2.0 * tail))
            result["positive_count"] = positives
            result["negative_count"] = negatives
    return result


def run(paths: List[Path], output: Path) -> Dict[str, Any]:
    records: Dict[int, Dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("control_protocol", {})
        seed = int(protocol.get("model_seed", -1))
        if seed < 0:
            raise ValueError(f"Missing model_seed in {path}")
        if seed in records:
            raise ValueError(f"Duplicate seed {seed}: {path}")
        records[seed] = {
            "evaluation_file": str(path),
            "checkpoint": payload.get("checkpoint"),
            "val_threshold": payload.get("val_threshold"),
            "targets": payload.get("targets", {}),
        }

    seeds = sorted(records)
    result: Dict[str, Any] = {
        "schema": "p0-fixed-random-nonsemantic-summary-v1",
        "seeds": seeds,
        "complete_expected_seed_set": seeds == EXPECTED_SEEDS,
        "runs": records,
        "targets": {},
    }
    for dataset in ("inria", "massachusetts"):
        values: Dict[int, float] = {}
        for seed, record in records.items():
            target = record["targets"].get(dataset, {})
            honest = target.get("honest", {})
            if "iou" not in honest:
                raise ValueError(f"No source-only IoU for {dataset}, seed {seed}")
            values[seed] = float(honest["iou"]) * 100.0
        target_summary: Dict[str, Any] = {
            "per_seed_iou_percent": {str(seed): values[seed] for seed in sorted(values)},
            **mean_std([values[s] for s in sorted(values)]),
        }
        baseline = {s: BASELINE_NONE[s][dataset] for s in values if s in BASELINE_NONE}
        if baseline:
            target_summary["paired_vs_visual_none"] = paired_summary(values, baseline)
        cmce = (CMCE_INRIA if dataset == "inria" else CMCE_MASSACHUSETTS)
        cmce = {s: cmce[s] * 100.0 for s in values if s in cmce}
        if cmce:
            target_summary["canonical_cmce_source_validation"] = {
                "per_seed_iou_percent": {str(s): cmce[s] for s in sorted(cmce)},
                **mean_std([cmce[s] for s in sorted(cmce)]),
            }
            target_summary["paired_random_minus_canonical_cmce"] = paired_summary(
                {s: values[s] for s in values if s in cmce}, cmce
            )
        result["targets"][dataset] = target_summary

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "seeds": seeds}, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-json", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.eval_json, args.output)


if __name__ == "__main__":
    main()
