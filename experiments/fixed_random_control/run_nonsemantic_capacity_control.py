#!/usr/bin/env python
"""Train an exact-parameter non-semantic CMCE control.

The control keeps the canonical CMCE architecture, optimizer, source split,
augmentation, seed schedule, and 28,761,129 trainable parameters unchanged.
It replaces *every* Janus feature seen during source training, source
validation, and source testing with one fixed random 2048-D vector.  The
vector is generated from a recorded CPU seed and scaled only to the mean L2
norm of WHU-train unified embeddings; no description, target image, or target
label determines its direction.

This is a training-time capacity control, not an inference substitution.  It
tests whether the transfer gain survives after per-image text content is
removed while the additional CMCE branch capacity is held exactly fixed.

Example::

    python experiments/fixed_random_control/run_nonsemantic_capacity_control.py --seed 42 --gpu 0

Append ``--dry-run`` for a software and memory smoke test (one batch, one
forward pass, no optimizer step).  Outputs go below
``results/fixed_random_control/seed<seed>``; an existing checkpoint is never
overwritten silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.cleaned_unified_segmentation_model import CleanedUnifiedTrainer  # noqa: E402
from train import create_cleaned_model, create_data_loaders, set_seed  # noqa: E402


CANONICAL_TOTAL_PARAMETERS = 28_761_129
DEFAULT_VECTOR_SEED = 20260902


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous().float()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


class FixedTextLoader:
    """DataLoader view that replaces both compatibility feature keys.

    The underlying unified Janus files are still used to establish exact image
    coverage/order and tensor shape.  Their numerical content is never passed
    to CMCE after replacement.
    """

    def __init__(self, loader: Any, vector: torch.Tensor):
        self.loader = loader
        self.vector = vector.detach().cpu().float().reshape(-1)

    @property
    def dataset(self) -> Any:
        return self.loader.dataset

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.loader:
            if not isinstance(batch, dict):
                raise TypeError(f"Expected dict batch, got {type(batch).__name__}")
            reference = batch.get("text_feature", batch.get("text_features"))
            if not isinstance(reference, torch.Tensor):
                raise RuntimeError("The CMCE loader did not yield text features.")
            if reference.shape[-1] != self.vector.numel():
                raise ValueError(
                    f"Feature dim {reference.shape[-1]} does not match fixed vector {self.vector.numel()}"
                )
            view_shape = [1] * (reference.ndim - 1) + [self.vector.numel()]
            replacement = self.vector.view(*view_shape).expand(*reference.shape).clone()
            out = dict(batch)
            out["text_feature"] = replacement
            out["text_features"] = replacement
            yield out


def source_mean_norm(data_root: Path) -> float:
    """Use only the source TRAIN feature norm to preserve input scale."""
    cache = data_root / "whu_building" / "train" / "unified_janus_features" / "batch_features.pt"
    if not cache.exists():
        raise FileNotFoundError(f"Required source feature cache not found: {cache}")
    try:
        payload = torch.load(cache, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache, map_location="cpu")
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 2048:
        raise RuntimeError("Expected [N,2048] tensor at WHU train unified batch_features.pt")
    return float(features.float().norm(dim=1).mean().item())


def make_fixed_random_vector(seed: int, norm: float, dimension: int = 2048) -> torch.Tensor:
    """Make a version-independent vector using SHA-256/Box--Muller draws.

    ``torch.randn`` bit streams differ across PyTorch versions.  A small
    counter-based construction keeps the control vector byte-identical across
    CUDA/PyTorch versions; the saved ``fixed_vector.pt`` beside each checkpoint
    is the final authority.
    """
    values = []
    for index in range(0, dimension, 2):
        raw_u1 = hashlib.sha256(f"cmce-p0:{seed}:{index}:u1".encode("ascii")).digest()[:8]
        raw_u2 = hashlib.sha256(f"cmce-p0:{seed}:{index}:u2".encode("ascii")).digest()[:8]
        u1 = (int.from_bytes(raw_u1, "big") + 0.5) / float(1 << 64)
        u2 = (int.from_bytes(raw_u2, "big") + 0.5) / float(1 << 64)
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        values.extend((radius * math.cos(angle), radius * math.sin(angle)))
    scale = float(norm) / math.sqrt(math.fsum(value * value for value in values))
    return torch.tensor([value * scale for value in values[:dimension]], dtype=torch.float32).contiguous()


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("fixed_random_control")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(payload)), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(
    path: Path,
    data_root: Path,
    workers: Optional[int],
    batch_size: Optional[int],
    epochs: Optional[int],
    pin_memory: Optional[bool],
) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(path)
    config["data"]["dataset"] = "whu_building"
    config["data"]["root_dir"] = str(data_root)
    config.setdefault("datasets", {}).setdefault("whu_building", {})["root_dir"] = str(data_root / "whu_building")
    if workers is not None:
        config["data"]["num_workers"] = workers
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
        config["training"]["val_batch_size"] = batch_size
    if epochs is not None:
        config["training"]["num_epochs"] = epochs
    if pin_memory is not None:
        config.setdefault("hardware", {})["pin_memory"] = bool(pin_memory)
    return config


def device_for(gpu: Optional[int], force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu is None:
        return torch.device("cuda:0")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu} unavailable; visible count={torch.cuda.device_count()}")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(args: argparse.Namespace, config_path: Path, output_root: Path, data_root: Path) -> Dict[str, Any]:
    run_dir = output_root / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pth"
    if best_path.exists() and not args.allow_existing:
        raise FileExistsError(
            f"{best_path} already exists. Refusing to overwrite an experiment; use --allow-existing only deliberately."
        )

    # The model seed governs initialization and loader shuffling.  The fixed
    # vector uses a separate recorded seed, so the control is identical across
    # model seeds and cannot be selected post hoc.
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(config_path, data_root, args.workers, args.batch_size, args.epochs, args.pin_memory)
    source_norm = source_mean_norm(data_root)
    fixed_vector = make_fixed_random_vector(args.vector_seed, source_norm)
    device = device_for(args.gpu, args.cpu)
    logger = setup_logger(run_dir)

    protocol = {
        "schema": "p0-fixed-random-nonsemantic-control-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "control_description": (
            "Same CMCE architecture/training protocol, but every source train/val/test text tensor is replaced "
            "by one fixed random 2048-D vector."
        ),
        "semantic_input": "none; vector direction is sampled from a recorded random seed",
        "source_scale_only": "mean L2 norm of WHU train unified features",
        "source_feature_dir": "unified_janus_features",
        "source_text_dir": "unified_janus_texts",
        "model_seed": args.seed,
        "fixed_vector_seed": args.vector_seed,
        "fixed_vector_dim": int(fixed_vector.numel()),
        "fixed_vector_l2_norm": float(fixed_vector.norm().item()),
        "source_train_mean_l2_norm": source_norm,
        "fixed_vector_sha256_float32": tensor_sha256(fixed_vector),
        "fixed_vector_file": "fixed_vector.pt",
        "config_path": relative_or_absolute(config_path),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_wrapper_sha256": sha256_file(ROOT / "models" / "cleaned_unified_segmentation_model.py"),
        "cmce_module_sha256": sha256_file(ROOT / "alignment" / "cmce.py"),
        "trainer": "CleanedUnifiedTrainer",
        "device": str(device),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "requested_epochs": int(config["training"]["num_epochs"]),
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": bool(config.get("hardware", {}).get("pin_memory", True)),
        "persistent_workers": bool(
            config.get("hardware", {}).get("persistent_workers", True)
            and int(config["data"]["num_workers"]) > 0
        ),
    }
    write_json(run_dir / "protocol.json", protocol)
    torch.save(fixed_vector, run_dir / "fixed_vector.pt")
    logger.info("fixed-random non-semantic capacity control")
    logger.info("run directory: %s", run_dir)
    logger.info("device=%s model_seed=%s fixed_vector_seed=%s", device, args.seed, args.vector_seed)
    logger.info("fixed vector: dim=%s norm=%.6f sha256=%s", fixed_vector.numel(), fixed_vector.norm(), protocol["fixed_vector_sha256_float32"])

    train_loader, val_loader, test_loader = create_data_loaders(
        config, alignment_type="cmce", text_type="janus", unified_prompt=True
    )
    fixed_train = FixedTextLoader(train_loader, fixed_vector)
    fixed_val = FixedTextLoader(val_loader, fixed_vector)
    fixed_test = FixedTextLoader(test_loader, fixed_vector)
    model = create_cleaned_model(config, "cmce", "janus").to(device)
    model_info = model.get_model_info()
    total_parameters = int(model_info["params_total"])
    if total_parameters != CANONICAL_TOTAL_PARAMETERS:
        raise AssertionError(
            f"Parameter mismatch: got {total_parameters:,}, expected {CANONICAL_TOTAL_PARAMETERS:,}"
        )
    protocol["model_info"] = model_info
    protocol["parameter_match"] = {
        "expected_total": CANONICAL_TOTAL_PARAMETERS,
        "observed_total": total_parameters,
        "matched": True,
    }
    write_json(run_dir / "protocol.json", protocol)
    logger.info("exact parameter match: %s trainable parameters", f"{total_parameters:,}")

    # A real forward pass validates that both text compatibility keys are
    # replaced and that the fixed vector reaches CMCE with the expected shape.
    first_batch = next(iter(fixed_train))
    with torch.no_grad():
        model.eval()
        logits = model(first_batch["image"].to(device), first_batch["text_feature"].to(device))
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits in pre-training non-semantic control smoke check")
    smoke = {
        "image_shape": list(first_batch["image"].shape),
        "replaced_text_shape": list(first_batch["text_feature"].shape),
        "replaced_text_rows_identical": bool(
            torch.allclose(first_batch["text_feature"], fixed_vector.view(1, -1).expand_as(first_batch["text_feature"]))
        ),
        "logits_shape": list(logits.shape),
        "logits_finite": True,
    }
    if device.type == "cuda":
        smoke["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    protocol["smoke_check"] = smoke
    write_json(run_dir / "protocol.json", protocol)
    logger.info("smoke check: image=%s text=%s logits=%s", smoke["image_shape"], smoke["replaced_text_shape"], smoke["logits_shape"])

    if args.dry_run:
        summary = {**protocol, "status": "dry_run_passed"}
        write_json(run_dir / "summary.json", summary)
        logger.info("dry run complete; no optimizer step or checkpoint was written")
        return summary

    trainer = CleanedUnifiedTrainer(model, device, config, logger)
    train_history, val_history = trainer.train(
        fixed_train, fixed_val, int(config["training"]["num_epochs"]), str(run_dir)
    )
    if not best_path.exists():
        raise RuntimeError("Training completed without a best_model.pth checkpoint.")
    source_test = trainer.evaluate_test_set(fixed_test)
    try:
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location="cpu")
    checkpoint["p0_nonsemantic_control"] = protocol
    checkpoint["source_test_nonsemantic"] = source_test
    torch.save(checkpoint, best_path)
    summary = {
        **protocol,
        "status": "completed",
        "best_epoch": int(trainer.best_epoch),
        "best_val_iou": float(trainer.best_val_iou),
        "best_threshold": float(trainer.best_eval_threshold),
        "source_test": source_test,
        "train_history": dict(train_history),
        "val_history": dict(val_history),
        "checkpoint": relative_or_absolute(best_path),
        "checkpoint_sha256": sha256_file(best_path),
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("completed: best val IoU=%.6f threshold=%.2f source test IoU=%.6f", trainer.best_val_iou, trainer.best_eval_threshold, source_test["iou"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cmce_v5_canonical.yaml")
    parser.add_argument("--data-root", default="data", help="directory holding whu_building/")
    parser.add_argument("--output-root", default="results/fixed_random_control")
    parser.add_argument("--seed", type=int, required=True, help="CMCE initialization/data-order seed")
    parser.add_argument("--vector-seed", type=int, default=DEFAULT_VECTOR_SEED)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="force CPU (software check only)")
    parser.add_argument("--epochs", type=int, default=None, help="override canonical epochs only when explicitly intended")
    parser.add_argument("--batch-size", type=int, default=None, help="override canonical batch size only when explicitly intended")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        default=None,
        help="disable DataLoader pinned-memory threads (useful on shared Linux nodes)",
    )
    parser.add_argument("--dry-run", action="store_true", help="instantiate/load one batch/forward pass without training")
    parser.add_argument("--allow-existing", action="store_true", help="allow an existing run directory; never silently overwrites a checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    os.chdir(ROOT)
    summary = run(args, config_path, output_root, data_root)
    print(json.dumps({"status": summary["status"], "seed": args.seed, "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
