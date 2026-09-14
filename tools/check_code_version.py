#!/usr/bin/env python
"""Verify that the model and training code match the version that produced the paper's results.

Fingerprints are the first 16 hex digits of the SHA-256 of each file with CR characters removed
(so Windows and Unix checkouts agree). They were pinned on 2026-06-29, before the paper's
experiments were run. Exit code 0 means every listed file matches.

Example:
    python tools/check_code_version.py
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PINNED = {
    "ablation_cmce.py": "a9cb2c74467f2e68",
    "alignment/__init__.py": "e3b0c44298fc1c14",
    "alignment/cmce.py": "2b724547afcc195e",
    "alignment/daca.py": "ea1ae419bca100d9",
    "alignment/dynamic_alignment.py": "43535f7de7753c5f",
    "alignment/generative_alignment.py": "e450bbdcd25f1db4",
    "alignment/simple_alignment.py": "24f180dcc33f6d4e",
    "configs/cmce_v5_canonical.yaml": "861310633cd2665c",
    "models/__init__.py": "e3b0c44298fc1c14",
    "models/backbones/__init__.py": "f7b1e813a3a6465b",
    "models/backbones/unet.py": "f872c318b191489b",
    "models/cleaned_unified_segmentation_model.py": "81a41b80315437c0",
    "train.py": "0f4289f5e211117e",
    "utils/__init__.py": "e3b0c44298fc1c14",
    "utils/transforms.py": "7249ba111ed868fe",
    "utils/unified_data_manager.py": "839cbe9ea6aef06b",
    "utils/unified_loss.py": "12d5830c95658466",
}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r", b"")).hexdigest()[:16]


def main():
    failures = 0
    for rel, want in PINNED.items():
        path = ROOT / rel
        if not path.exists():
            print(f"MISSING  {rel}")
            failures += 1
            continue
        have = fingerprint(path)
        if have != want:
            print(f"CHANGED  {rel}  (have {have}, pinned {want})")
            failures += 1
    if failures:
        print(f"{failures} file(s) differ from the paper version")
        return 1
    print(f"all {len(PINNED)} files match the paper version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
