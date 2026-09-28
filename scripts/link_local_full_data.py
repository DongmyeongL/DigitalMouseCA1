#!/usr/bin/env python3
"""Link the existing 39,112-cell runtime packages into the public layout."""

import argparse
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "data_registry.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO.parent,
        help="Root of the existing hippocampus_seeg_modeling workspace",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    for name, package in registry["packages"].items():
        source = (args.source_root.resolve() / package["local_source"]).resolve()
        destination = REPO / package["install_dir"]
        if not source.is_dir():
            raise FileNotFoundError(f"{name}: missing source directory: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() == source:
                print(f"[OK] {name}: already linked")
                continue
            if not args.replace:
                raise RuntimeError(f"{destination} already links elsewhere; use --replace")
            destination.unlink()
        elif destination.exists():
            raise RuntimeError(f"{destination} exists and is not a symlink")
        relative = os.path.relpath(source, destination.parent)
        destination.symlink_to(relative, target_is_directory=True)
        print(f"[LINK] {name}: {destination} -> {relative}")

    print("Full model data are connected. Run scripts/verify_full_data.py next.")


if __name__ == "__main__":
    main()

