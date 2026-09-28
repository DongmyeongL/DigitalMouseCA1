#!/usr/bin/env python3
"""Validate the files and biological dimensions required by the full model."""

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "data_registry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksums", action="store_true", help="Hash all 13.5 GB (slower)")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    failures = []
    lock = {"schema_version": 1, "release": registry["release"], "files": {}}

    for name, package in registry["packages"].items():
        directory = REPO / package["install_dir"]
        print(f"[{name}] {directory}")
        if not directory.is_dir():
            failures.append(f"missing package: {name}")
            continue
        lock["files"][name] = {}
        for relative, expected_bytes in package["required_files"].items():
            path = directory / relative
            if not path.is_file():
                failures.append(f"{name}: missing {relative}")
                continue
            actual = path.stat().st_size
            if actual != expected_bytes:
                failures.append(
                    f"{name}/{relative}: {actual} bytes, expected {expected_bytes}"
                )
                continue
            record = {"bytes": actual}
            if args.checksums:
                record["sha256"] = sha256(path)
            lock["files"][name][relative] = record
        print(f"  validated {len(lock['files'][name])} required files")

    if failures:
        print("\nVALIDATION FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    lock_path = REPO / "data" / "full_data.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    model = registry["model"]
    print("\nFULL MODEL DATA READY")
    print(
        f"  {model['cells']:,} cells | {model['compartments']:,} compartments | "
        f"{model['edges']:,} edges | {model['synapses']:,} synapses | "
        f"{model['electrodes']} electrodes"
    )
    print(f"  lock manifest: {lock_path}")


if __name__ == "__main__":
    main()

