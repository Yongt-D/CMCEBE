#!/usr/bin/env python
"""Evaluate WHU-trained checkpoints on a target domain under the paper's threshold protocols.

Each checkpoint is run once over the full target split. Pixel-pooled tp/fp/fn are accumulated
on a 0.05-0.95 threshold grid (step 0.01), and four protocols are read from that single sweep:

  source   the threshold selected on WHU validation during training, stored in the checkpoint
           as ``best_threshold``. No target label is used. This is the primary protocol.
  window   best IoU within [source - 0.15, source + 0.15] on the target split. Uses target
           labels; reported only as a sensitivity analysis.
  oracle   best IoU over the whole grid. Uses target labels; an upper bound.
  fixed05  a fixed threshold of 0.5.

Example (WHU -> Inria, CMCE, five seeds):

    python tools/evaluate_transfer.py --dataset inria \
        --checkpoints weights/cmce_whu_seed0.pth weights/cmce_whu_seed42.pth \
                      weights/cmce_whu_seed123.pth weights/cmce_whu_seed456.pth \
                      weights/cmce_whu_seed666.pth \
        --out results/transfer_inria_cmce.json

The alignment type is read from each checkpoint (``alignment_type``); --alignment overrides it.
Text-conditioned models refuse to run unless every image has a text feature, so a missing
feature directory cannot silently turn the evaluation into a text-free one.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import create_cleaned_model  # noqa: E402
from utils.transforms import BuildingExtractionTransforms  # noqa: E402
from utils.unified_data_manager import UnifiedDataManager  # noqa: E402

GRID = np.round(np.arange(0.05, 0.9501, 0.01), 3)
DEFAULT_CONFIGS = {
    "whu_building": "configs/cmce_v5_canonical.yaml",
    "inria": "configs/cmce_v5_canonical_inria.yaml",
    "massachusetts": "configs/cmce_v5_canonical_mass.yaml",
}
PROTOCOLS = {
    "source": "threshold selected on WHU validation (checkpoint best_threshold); no target labels",
    "window": "best IoU within [source-0.15, source+0.15] on the target split; uses target labels",
    "oracle": "best IoU over the full 0.05-0.95 grid; uses target labels",
    "fixed05": "fixed threshold 0.5",
}


def load_config(path, dataset, data_root):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["data"]["dataset"] = dataset
    if data_root is not None:
        config["data"]["root_dir"] = str(data_root)
        config.setdefault("datasets", {}).setdefault(dataset, {})["root_dir"] = str(data_root / dataset)
    return config


def make_loader(config, dataset, split, alignment, batch_size, workers, text_dir, text_sub):
    manager = UnifiedDataManager(config)
    transform = BuildingExtractionTransforms(
        phase="test", image_size=config["data"]["image_size"], augmentation_config={}
    )
    loader = manager.get_dataloader(
        dataset_name=dataset, split=split, batch_size=batch_size, shuffle=False,
        num_workers=workers, alignment_type=alignment, text_type="janus", transform=transform,
        text_feature_subdir=text_dir, text_subdir=text_sub,
    )
    if alignment != "none":
        samples = loader.dataset.samples
        missing = [s["name"] for s in samples if not s.get("text_path")]
        if not loader.dataset.load_text or missing:
            raise SystemExit(
                f"{dataset}/{split}: text features missing in '{text_dir}' "
                f"({len(missing)} of {len(samples)} images); refusing a text-free evaluation"
            )
    return loader


@torch.no_grad()
def sweep(model, loader, device, use_text):
    grid = torch.tensor(GRID, device=device).view(-1, 1)
    tp = torch.zeros(len(GRID), dtype=torch.float64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    n_images = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).bool().view(1, -1)
        logits = model(image, batch["text_feature"].to(device)) if use_text else model(image)
        pred = torch.sigmoid(logits).view(1, -1) > grid
        tp += (pred & label).sum(1).double()
        fp += (pred & ~label).sum(1).double()
        fn += (~pred & label).sum(1).double()
        n_images += image.shape[0]
    iou = (tp / (tp + fp + fn).clamp(min=1)).cpu().numpy()
    precision = (tp / (tp + fp).clamp(min=1)).cpu().numpy()
    recall = (tp / (tp + fn).clamp(min=1)).cpu().numpy()
    return iou, precision, recall, n_images


def metrics_at(index, iou, precision, recall):
    return {"threshold": float(GRID[index]), "iou": float(iou[index]),
            "precision": float(precision[index]), "recall": float(recall[index])}


def summarize(runs):
    summary = {}
    for alignment in sorted({r["alignment"] for r in runs}):
        group = [r for r in runs if r["alignment"] == alignment]
        summary[alignment] = {"n": len(group), "seeds": [r["seed"] for r in group]}
        for protocol in PROTOCOLS:
            values = np.array([r[protocol]["iou"] * 100 for r in group])
            summary[alignment][protocol] = {
                "iou_percent": [round(float(v), 4) for v in values],
                "mean": float(values.mean()),
                "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset", default="inria", choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default=None, help="defaults to the canonical config for --dataset")
    parser.add_argument("--data-root", default=None,
                        help="directory holding <dataset>/<split>/...; defaults to data/ in the repository")
    parser.add_argument("--alignment", default=None, choices=["none", "simple", "dynamic", "generative", "cmce"],
                        help="override the alignment type stored in the checkpoints")
    parser.add_argument("--text-dir", default="unified_janus_features")
    parser.add_argument("--text-sub", default="unified_janus_texts")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoints = [Path(p).resolve() for p in args.checkpoints]
    out_path = Path(args.out).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else None
    config_path = Path(args.config).resolve() if args.config else ROOT / DEFAULT_CONFIGS[args.dataset]
    os.chdir(ROOT)  # the canonical configs use repository-relative data paths
    config = load_config(config_path, args.dataset, data_root)
    device = torch.device(args.device)

    report = {"dataset": args.dataset, "split": args.split, "text_dir": args.text_dir,
              "config": config_path.name, "grid": [float(t) for t in GRID],
              "metric": "global pixel-pooled tp/fp/fn over the split", "protocols": PROTOCOLS, "runs": []}
    loaders = {}
    for ckpt_path in checkpoints:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        alignment = args.alignment or payload.get("alignment_type")
        if alignment is None:
            raise SystemExit(f"{ckpt_path.name}: no alignment_type stored; pass --alignment")
        if "best_threshold" not in payload:
            raise SystemExit(f"{ckpt_path.name}: no best_threshold stored; the source protocol needs it")
        source_threshold = float(payload["best_threshold"])

        if alignment not in loaders:
            loaders[alignment] = make_loader(config, args.dataset, args.split, alignment,
                                             args.batch_size, args.workers, args.text_dir, args.text_sub)
        model = create_cleaned_model(config, alignment, "janus").to(device)
        model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
        model.eval()

        iou, precision, recall, n_images = sweep(model, loaders[alignment], device, alignment != "none")
        lo, hi = max(0.05, source_threshold - 0.15), min(0.95, source_threshold + 0.15)
        window = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
        row = {
            "checkpoint": ckpt_path.name,
            "alignment": alignment,
            "seed": payload.get("seed"),
            "source_threshold": source_threshold,
            "n_images": n_images,
            "source": metrics_at(int(np.argmin(np.abs(GRID - source_threshold))), iou, precision, recall),
            "window": metrics_at(int(np.argmax(np.where(window, iou, -1))), iou, precision, recall),
            "oracle": metrics_at(int(np.argmax(iou)), iou, precision, recall),
            "fixed05": metrics_at(int(np.argmin(np.abs(GRID - 0.5))), iou, precision, recall),
        }
        report["runs"].append(row)
        print("%-24s %-7s thr=%.2f | source %.2f | window %.2f | oracle %.2f | fixed0.5 %.2f"
              % (ckpt_path.name, alignment, source_threshold, row["source"]["iou"] * 100,
                 row["window"]["iou"] * 100, row["oracle"]["iou"] * 100, row["fixed05"]["iou"] * 100),
              flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        report["summary"] = summarize(report["runs"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for alignment, block in report["summary"].items():
        std = block["source"]["std_sample"]
        print("%-7s n=%d  source-only IoU %.2f%s" % (alignment, block["n"], block["source"]["mean"],
                                                   "" if std is None else " +- %.2f" % std))
    print("saved " + str(out_path))


if __name__ == "__main__":
    main()
