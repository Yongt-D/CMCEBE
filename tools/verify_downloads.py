#!/usr/bin/env python
"""Verify downloaded weights and text-data archives against the published SHA-256 checksums.

Examples:
    python tools/verify_downloads.py --sums checksums/weights_SHA256SUMS.txt --dir weights
    python tools/verify_downloads.py --sums checksums/text_data_SHA256SUMS.txt --dir downloads
"""
import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sums", required=True, help="checksum file with lines '<sha256>  <file name>'")
    parser.add_argument("--dir", required=True, help="directory holding the downloaded files")
    args = parser.parse_args()

    failures = 0
    for line in Path(args.sums).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = Path(args.dir) / name.strip()
        if not path.exists():
            print(f"MISSING   {name}")
            failures += 1
        elif sha256(path) != expected:
            print(f"MISMATCH  {name}")
            failures += 1
        else:
            print(f"ok        {name}")
    if failures:
        print(f"{failures} file(s) missing or corrupted")
        return 1
    print("all files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
