#!/usr/bin/env python
"""Evaluate a trained fixed-random CMCE control on unseen target domains.

The checkpoint's WHU validation threshold is used unchanged for the source-only
transfer number.  Target labels are accumulated only for reporting the IoU
curve and explicitly labelled diagnostics; they are never used to select a
model or threshold.  Text is replaced by the same fixed vector used for
training, so this is a trained-control evaluation rather than an
inference-only substitution.

Example::

    python experiments/fixed_random_control/evaluate_nonsemantic_capacity.py \
        --checkpoint results/fixed_random_control/seed42/best_model.pth \
        --output results/fixed_random_control/seed42/evaluation_inria_mass.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import create_cleaned_model  # noqa: E402
from utils.transforms import BuildingExtractionTransforms  # noqa: E402
from utils.unified_data_manager import UnifiedDataManager  # noqa: E402
from run_nonsemantic_capacity_control import FixedTextLoader, make_fixed_random_vector, tensor_sha256  # noqa: E402


GRID = np.round(np.arange(0.05, 0.9501, 0.01), 3)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def target_config(base_config: Dict[str, Any], dataset: str, data_root: Path, workers: int) -> Dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["data"]["dataset"] = dataset
    config["data"]["num_workers"] = workers
    config["data"]["root_dir"] = str(data_root)
    config.setdefault("datasets", {})[dataset] = {
        "root_dir": str(data_root / dataset),
        "train_samples": -1,
        "description": dataset,
    }
    return config


def make_target_loader(config: Dict[str, Any], dataset: str, batch_size: int, workers: int):
    tf = BuildingExtractionTransforms(
        phase="test", image_size=int(config["data"]["image_size"]), augmentation_config={}
    )
    dm = UnifiedDataManager(config)
    return dm.get_dataloader(
        dataset_name=dataset,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        alignment_type="cmce",
        text_type="janus",
        transform=tf,
        text_feature_subdir="unified_janus_features",
        text_subdir="unified_janus_texts",
    )


@torch.no_grad()
def sweep(model: torch.nn.Module, loader: Iterable[Dict[str, Any]], device: torch.device):
    thresholds = torch.tensor(GRID, device=device).view(-1, 1)
    tp = torch.zeros(len(GRID), dtype=torch.float64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    n_images = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["label"].to(device, non_blocking=True).bool()
        text = batch["text_feature"].to(device, non_blocking=True)
        probability = torch.sigmoid(model(image, text))
        flat_prob = probability.reshape(1, -1)
        flat_mask = mask.reshape(1, -1)
        predicted = flat_prob > thresholds
        tp += (predicted & flat_mask).sum(1).double()
        fp += (predicted & ~flat_mask).sum(1).double()
        fn += (~predicted & flat_mask).sum(1).double()
        n_images += int(image.shape[0])
    iou = (tp / (tp + fp + fn).clamp_min(1)).cpu().numpy()
    precision = (tp / (tp + fp).clamp_min(1)).cpu().numpy()
    recall = (tp / (tp + fn).clamp_min(1)).cpu().numpy()
    return iou, precision, recall, n_images


def metric_at(iou: np.ndarray, precision: np.ndarray, recall: np.ndarray, threshold: float) -> Dict[str, float]:
    index = int(np.argmin(np.abs(GRID - float(threshold))))
    return {
        "threshold": float(GRID[index]),
        "iou": float(iou[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
    }


def metric_best(iou: np.ndarray, precision: np.ndarray, recall: np.ndarray, allowed: Optional[np.ndarray] = None) -> Dict[str, float]:
    if allowed is None:
        allowed = np.ones_like(iou, dtype=bool)
    masked = np.where(allowed, iou, -1.0)
    index = int(np.argmax(masked))
    return {
        "threshold": float(GRID[index]),
        "iou": float(iou[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
    }


def run(args: argparse.Namespace, checkpoint_path: Path, config_path: Path, data_root: Path, out_path: Path) -> Dict[str, Any]:
    protocol_path = checkpoint_path.parent / "protocol.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    protocol = checkpoint.get("p0_nonsemantic_control")
    if protocol is None and protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol:
        raise RuntimeError("Checkpoint has no p0_nonsemantic_control protocol metadata")
    parameter_match = protocol.get("parameter_match", {})
    if parameter_match and not parameter_match.get("matched", False):
        raise RuntimeError("Checkpoint protocol does not certify the canonical parameter match")
    if parameter_match and int(parameter_match.get("observed_total", -1)) != 28_761_129:
        raise RuntimeError("Checkpoint parameter count differs from the canonical CMCE control")
    vector_path = checkpoint_path.parent / protocol.get("fixed_vector_file", "fixed_vector.pt")
    if vector_path.exists():
        try:
            vector = torch.load(vector_path, map_location="cpu", weights_only=False)
        except TypeError:
            vector = torch.load(vector_path, map_location="cpu")
        vector = vector.detach().cpu().float().reshape(-1)
    else:
        vector = make_fixed_random_vector(int(protocol["fixed_vector_seed"]), float(protocol["source_train_mean_l2_norm"]))
    expected_hash = protocol.get("fixed_vector_sha256_float32")
    if expected_hash and tensor_sha256(vector) != expected_hash:
        raise RuntimeError("Fixed-vector hash does not match the checkpoint protocol metadata")

    with config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device(f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    val_threshold = float(checkpoint.get("best_threshold", protocol.get("best_threshold", 0.5)))

    output: Dict[str, Any] = {
        "schema": "p0-fixed-random-nonsemantic-evaluation-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "control_protocol": protocol,
        "threshold_protocol": "honest uses WHU source-validation best_threshold; window and oracle are diagnostics only",
        "val_threshold": val_threshold,
        "device": str(device),
        "targets": {},
    }
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        config = target_config(base_config, dataset, data_root, args.workers)
        model = create_cleaned_model(config, "cmce", "janus").to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        loader = make_target_loader(config, dataset, args.batch_size, args.workers)
        wrapped = FixedTextLoader(loader, vector)
        iou, precision, recall, n_images = sweep(model, wrapped, device)
        lo, hi = max(0.05, val_threshold - 0.15), min(0.95, val_threshold + 0.15)
        window = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
        output["targets"][dataset] = {
            "split": "test",
            "text_feature_dir": "unified_janus_features (replaced by fixed vector)",
            "n_images": n_images,
            "honest": metric_at(iou, precision, recall, val_threshold),
            "paper_window_diagnostic": metric_best(iou, precision, recall, window),
            "oracle_diagnostic": metric_best(iou, precision, recall),
            "fixed_vector_rows_identical": True,
        }
        print(
            f"{dataset}: n={n_images} source-only IoU={output['targets'][dataset]['honest']['iou'] * 100:.3f}% "
            f"oracle IoU={output['targets'][dataset]['oracle_diagnostic']['iou'] * 100:.3f}%",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out_path}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/cmce_v5_canonical.yaml")
    parser.add_argument("--data-root", default="data", help="directory holding inria/ and massachusetts/")
    parser.add_argument("--datasets", default="inria,massachusetts")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    out_path = Path(args.output).resolve()
    os.chdir(ROOT)
    run(args, checkpoint_path, config_path, data_root, out_path)


if __name__ == "__main__":
    main()
