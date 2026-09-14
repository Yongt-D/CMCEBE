#!/usr/bin/env python
"""Check whether a CMCE checkpoint was really trained with text.

When the text-feature directory is missing during training, the model silently trains only its
U-Net fallback path and still logs a normal IoU, so recorded metrics cannot reveal it. Evaluating
the same checkpoint with and without text can:

    trained with text   high IoU with text, collapses without text
    trained without     collapses with text, high IoU without text

The CMCE geometry (number of anchors, iterations, ...) is taken from the configuration stored in
each checkpoint, so sweep checkpoints load strictly.

Example:
    python tools/audit_textless_checkpoint.py --text-dir text_features \
        --checkpoints checkpoints/whu_building/ablation/ablation_full_cmce_seed42/best_model.pth
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


@torch.no_grad()
def iou_at(model, loader, device, threshold, feed_text):
    tp = fp = fn = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).bool()
        text = batch["text_feature"].to(device) if feed_text else None
        pred = torch.sigmoid(model(image, text)) > threshold
        tp += float((pred & label).sum())
        fp += float((pred & ~label).sum())
        fn += float((~pred & label).sum())
    return tp / max(tp + fp + fn, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--config", default="configs/cmce_v5_canonical.yaml")
    p.add_argument("--dataset", default="whu_building")
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", default=None, help="directory holding <dataset>/<split>/...; defaults to data/")
    p.add_argument("--text-dir", default="unified_janus_features",
                   help="use the directory the checkpoint was trained with (text_features for ablation_cmce.py)")
    p.add_argument("--limit", type=int, default=300, help="first N images of the split; <= 0 uses all")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    checkpoints = [Path(c).resolve() for c in a.checkpoints]
    config_path = Path(a.config).resolve()
    data_root = Path(a.data_root).resolve() if a.data_root else None
    out_path = Path(a.out).resolve() if a.out else None
    os.chdir(ROOT)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["data"]["dataset"] = a.dataset
    if data_root is not None:
        config["data"]["root_dir"] = str(data_root)
        config.setdefault("datasets", {}).setdefault(a.dataset, {})["root_dir"] = str(data_root / a.dataset)
    device = torch.device(a.device)

    manager = UnifiedDataManager(config)
    transform = BuildingExtractionTransforms(phase="test", image_size=config["data"]["image_size"],
                                             augmentation_config={})
    loader = manager.get_dataloader(dataset_name=a.dataset, split=a.split, batch_size=a.bs, shuffle=False,
                                    num_workers=a.workers, alignment_type="cmce", text_type="janus",
                                    transform=transform, text_feature_subdir=a.text_dir,
                                    text_subdir="unified_janus_texts", sample_count=a.limit)
    dataset = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    if not dataset.load_text:
        raise SystemExit(f"text directory {a.text_dir} is missing for {a.dataset}/{a.split}")

    rows = []
    print("%-60s %-10s %-10s %s" % ("checkpoint", "with text", "no text", "verdict"))
    for ckpt in checkpoints:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = json.loads(json.dumps(config))
        stored_model = (payload.get("config") or {}).get("model")
        if stored_model:
            cfg["model"] = stored_model
        model = create_cleaned_model(cfg, "cmce", "janus").to(device)
        model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
        model.eval()
        threshold = float(payload.get("best_threshold", 0.5))
        with_text = iou_at(model, loader, device, threshold, True)
        no_text = iou_at(model, loader, device, threshold, False)
        if with_text >= 0.5 > no_text:
            verdict = "TRAINED WITH TEXT"
        elif no_text >= 0.5 > with_text:
            verdict = "TRAINED WITHOUT TEXT"
        else:
            verdict = "INCONCLUSIVE (rerun with --limit 0)"
        rows.append({"checkpoint": str(ckpt), "threshold": threshold, "iou_with_text": with_text,
                     "iou_no_text": no_text, "verdict": verdict})
        print("%-60s %-10.4f %-10.4f %s" % (str(ckpt)[-60:], with_text, no_text, verdict), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"dataset": a.dataset, "split": a.split, "text_dir": a.text_dir,
                                        "limit": a.limit, "runs": rows}, indent=2), encoding="utf-8")
        print("saved " + str(out_path))


if __name__ == "__main__":
    main()
