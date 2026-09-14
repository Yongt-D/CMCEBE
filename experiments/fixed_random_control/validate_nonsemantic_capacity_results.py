#!/usr/bin/env python
"""Validate completed fixed-random control artifacts without changing them.

The checker is deliberately conservative: a run is accepted only when its
summary, checkpoint, protocol, fixed-vector hash, exact parameter certificate,
and both target evaluation records are present and internally consistent.

Example::

    python experiments/fixed_random_control/validate_nonsemantic_capacity_results.py \
        --run-dir results/fixed_random_control/seed0 results/fixed_random_control/seed42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch


EXPECTED_PARAMETERS = 28_761_129
EXPECTED_TARGETS = {"inria", "massachusetts"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous().float()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def same_threshold(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(run_dir: Path, evaluation_name: str) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    protocol_path = run_dir / "protocol.json"
    checkpoint_path = run_dir / "best_model.pth"
    vector_path = run_dir / "fixed_vector.pt"
    evaluation_path = run_dir / evaluation_name
    for path in (summary_path, protocol_path, checkpoint_path, vector_path, evaluation_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = load(summary_path)
    protocol = load(protocol_path)
    evaluation = load(evaluation_path)
    if summary.get("status") != "completed":
        raise RuntimeError(f"{run_dir}: summary status is {summary.get('status')!r}")
    if protocol.get("schema") != "p0-fixed-random-nonsemantic-control-v1":
        raise RuntimeError(f"{run_dir}: unexpected protocol schema")
    match = protocol.get("parameter_match", {})
    if not match.get("matched") or int(match.get("observed_total", -1)) != EXPECTED_PARAMETERS:
        raise RuntimeError(f"{run_dir}: exact parameter certificate failed")

    try:
        vector = torch.load(vector_path, map_location="cpu", weights_only=False)
    except TypeError:
        vector = torch.load(vector_path, map_location="cpu")
    expected_hash = protocol.get("fixed_vector_sha256_float32")
    if expected_hash and tensor_sha256(vector) != expected_hash:
        raise RuntimeError(f"{run_dir}: fixed-vector hash mismatch")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != summary.get("checkpoint_sha256"):
        raise RuntimeError(f"{run_dir}: checkpoint hash mismatch")
    if evaluation.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"{run_dir}: evaluation checkpoint hash mismatch")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_threshold = checkpoint.get("best_threshold") if isinstance(checkpoint, dict) else None
    if not same_threshold(summary.get("best_threshold"), checkpoint_threshold):
        raise RuntimeError(f"{run_dir}: summary/checkpoint source threshold mismatch")
    if not same_threshold(evaluation.get("val_threshold"), checkpoint_threshold):
        raise RuntimeError(f"{run_dir}: evaluation/checkpoint source threshold mismatch")
    if protocol.get("model_seed") != summary.get("model_seed"):
        raise RuntimeError(f"{run_dir}: summary/protocol model seed mismatch")
    if set(evaluation.get("targets", {})) != EXPECTED_TARGETS:
        raise RuntimeError(f"{run_dir}: target evaluation set is incomplete")
    for dataset in EXPECTED_TARGETS:
        target = evaluation["targets"][dataset]
        if int(target.get("n_images", 0)) <= 0 or "honest" not in target:
            raise RuntimeError(f"{run_dir}: incomplete {dataset} evaluation")
        if not same_threshold(target["honest"].get("threshold"), checkpoint_threshold):
            raise RuntimeError(f"{run_dir}: {dataset} source-only threshold is not the source-validation threshold")
    return {
        "run_dir": str(run_dir),
        "seed": int(protocol["model_seed"]),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "inria_source_only_iou": evaluation["targets"]["inria"]["honest"]["iou"],
        "massachusetts_source_only_iou": evaluation["targets"]["massachusetts"]["honest"]["iou"],
        "status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluation-name", default="evaluation_inria_mass.json")
    args = parser.parse_args()
    records = [validate(path.resolve(), args.evaluation_name) for path in args.run_dir]
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
