#!/usr/bin/env python
"""Check that every image has a text feature before launching text-conditioned training.

If the text-feature directory an entry point reads is missing, the data loader returns no text
and CMCE silently trains its text-free U-Net path while still logging a normal-looking IoU.
Run this first; exit code 0 means every image in every checked split has a feature file.

Directory read by each entry point:
    ablation_cmce.py              text_features            (Prompt-A features; the released CMCE weights)
    train.py --unified-prompt     unified_janus_features   (unified features; the released None/Simple weights)
    train.py                      text_features

Examples:
    python tools/preflight_check.py --entry ablation_cmce
    python tools/preflight_check.py --entry train_unified --data-root /path/to/data
"""
import argparse
import os
import sys

ENTRY_TEXT_DIR = {
    "ablation_cmce": "text_features",
    "train_unified": "unified_janus_features",
    "train": "text_features",
}


def stems(directory, suffixes):
    return {os.path.splitext(n)[0] for n in os.listdir(directory)
            if n.endswith(suffixes) and n != "batch_features.pt"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry", choices=sorted(ENTRY_TEXT_DIR), default="ablation_cmce")
    p.add_argument("--text-dir", default=None, help="override the directory implied by --entry")
    p.add_argument("--dataset", default="whu_building")
    p.add_argument("--data-root", default="data")
    p.add_argument("--splits", default="train,val,test")
    a = p.parse_args()

    text_dir = a.text_dir or ENTRY_TEXT_DIR[a.entry]
    print(f"dataset={a.dataset}  entry={a.entry}  text directory={text_dir}")
    ok = True
    for split in a.splits.split(","):
        images = os.path.join(a.data_root, a.dataset, split, "images")
        features = os.path.join(a.data_root, a.dataset, split, text_dir)
        if not os.path.isdir(images):
            print(f"  {split:5s} FAIL  missing images directory: {images}")
            ok = False
            continue
        if not os.path.isdir(features):
            print(f"  {split:5s} FAIL  missing {features}: training would silently drop the text branch")
            ok = False
            continue
        image_stems = stems(images, (".tif", ".png", ".jpg"))
        missing = sorted(image_stems - stems(features, (".pt",)))
        if missing:
            print(f"  {split:5s} FAIL  {len(missing)} of {len(image_stems)} images lack a feature, e.g. {missing[:3]}")
            ok = False
        else:
            print(f"  {split:5s} ok    {len(image_stems)} images, all with features")

    if not ok:
        print("\nDo not train: provide the missing text features first.")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
