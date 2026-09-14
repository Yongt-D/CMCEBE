#!/usr/bin/env python
"""Measure the BCLR refinement trace of trained CMCE checkpoints and plot it.

For an evenly spaced subset of the WHU test split, the model's own ``get_evolution_trace`` records
the alignment score at each refinement iteration and the change of the text state between
iterations 1 and 2. The raw values are written to JSON; the two-panel figure matches the
convergence figure of the paper.

Example:
    python experiments/convergence_trace.py --seeds 42,123,456 \
        --out-json results/convergence_trace.json --out-figure results/convergence_analysis.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import create_cleaned_model  # noqa: E402
from utils.transforms import BuildingExtractionTransforms  # noqa: E402
from utils.unified_data_manager import UnifiedDataManager, custom_collate_fn  # noqa: E402


def make_loader(config, n_images, text_dir):
    dm = UnifiedDataManager(config)
    tf = BuildingExtractionTransforms(
        phase="test", image_size=config["data"]["image_size"], augmentation_config={}
    )
    full = dm.get_dataloader(
        dataset_name="whu_building", split="test", batch_size=1, shuffle=False, num_workers=0,
        alignment_type="cmce", text_type="janus", transform=tf,
        text_feature_subdir=text_dir, text_subdir="unified_janus_texts",
    ).dataset
    ids = sorted(set(np.linspace(0, len(full) - 1, n_images).astype(int).tolist()))
    return DataLoader(Subset(full, ids), batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=custom_collate_fn)


def collect(config, checkpoint, data_loader, device, seed):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = create_cleaned_model(config, "cmce", "janus").to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    align, text_delta = [], []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader, 1):
            image = batch["image"].to(device)
            text = batch["text_feature"].to(device)
            _, feats = model.backbone.forward_with_features(image)
            scales = [feats[k] for k in model.cmce_feature_keys]
            trace = model.cmce_module.get_evolution_trace(scales, text)
            scores = trace["alignment_scores"]
            if len(scores) < 2:
                scores = scores + [scores[-1]] * (2 - len(scores))
            align.append([float(v) for v in scores[:2]])
            changes = trace["text_changes"]
            text_delta.append(float(changes[1]) if len(changes) > 1 else 0.0)
            if idx % 20 == 0:
                print(f"seed {seed}: {idx}/{len(data_loader)}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(align), np.asarray(text_delta)


def plot(records, seeds, n_images, out_figure):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.linewidth": 0.8,
    })
    colors = ["#4C78A8", "#E45756", "#54A24B", "#B279A2", "#F58518"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.15, 5.35), gridspec_kw={"height_ratios": [0.85, 1.15]}
    )

    x = np.array([1, 2])
    for color, seed in zip(colors, seeds):
        scores, _ = records[seed]
        ax1.errorbar(x, scores.mean(0), yerr=scores.std(0, ddof=1), marker="o",
                     lw=1.4, ms=5, capsize=2.5, color=color, label=f"seed {seed}")
    ax1.set_xlim(0.95, 2.05)
    ax1.set_xticks([1, 2])
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel("Iteration $t$")
    ax1.set_ylabel("Alignment score $s^{(t)}$")
    ax1.set_title("(a) Alignment score", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="lower right", fontsize=8, ncol=3)
    ax1.grid(True, ls="--", alpha=0.3)

    # The text changes are tiny; show them in units of 1e-8 (the JSON keeps raw values).
    for color, seed in zip(colors, seeds):
        ax2.hist(records[seed][1] * 1e8, bins=18, alpha=0.62, color=color, label=f"seed {seed}")
    ax2.set_xlabel(r"Text-state change $\|\Delta e_t^{(1)}\|$ ($\times 10^{-8}$)")
    ax2.set_ylabel(f"Count (out of {n_images} samples)")
    ax2.set_title("(b) Text change: iteration 1 to 2", loc="left", fontweight="bold")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(True, ls="--", alpha=0.3)
    ax2.ticklabel_format(axis="x", style="plain", useOffset=False)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.8, h_pad=1.0)
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=320, bbox_inches="tight")
    print(f"wrote {out_figure}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--config", default="configs/cmce_v5_canonical.yaml")
    p.add_argument("--data-root", default=None, help="directory holding whu_building/test/...; defaults to data/")
    p.add_argument("--text-dir", default="unified_janus_features")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-json", default="results/convergence_trace.json")
    p.add_argument("--out-figure", default="results/convergence_analysis.png")
    a = p.parse_args()

    weights_dir = Path(a.weights_dir).resolve()
    out_json = Path(a.out_json).resolve()
    out_figure = Path(a.out_figure).resolve()
    config_path = Path(a.config).resolve()
    data_root = Path(a.data_root).resolve() if a.data_root else None
    os.chdir(ROOT)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["data"]["dataset"] = "whu_building"
    if data_root is not None:
        config["data"]["root_dir"] = str(data_root)
        config.setdefault("datasets", {}).setdefault("whu_building", {})["root_dir"] = str(data_root / "whu_building")
    device = torch.device(a.device)
    seeds = [int(s) for s in a.seeds.split(",")]

    data_loader = make_loader(config, a.n_images, a.text_dir)
    records = {seed: collect(config, weights_dir / f"cmce_whu_seed{seed}.pth", data_loader, device, seed)
               for seed in seeds}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "dataset": "whu_building", "split": "test", "n_images": len(data_loader), "text_dir": a.text_dir,
        "alignment_scores_iter1_iter2": {str(s): records[s][0].tolist() for s in seeds},
        "text_change_iter1_to_iter2": {str(s): records[s][1].tolist() for s in seeds},
    }, indent=1), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    plot(records, seeds, len(data_loader), out_figure)


if __name__ == "__main__":
    main()
