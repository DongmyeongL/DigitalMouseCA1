#!/usr/bin/env python3
"""Download published full-model packages declared in data_registry.json."""

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "data_registry.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", nargs="+", default=["all"])
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    selected = list(registry["packages"]) if "all" in args.package else args.package
    unknown = sorted(set(selected) - set(registry["packages"]))
    if unknown:
        raise SystemExit(f"Unknown package(s): {', '.join(unknown)}")

    for name in selected:
        package = registry["packages"][name]
        url = package.get("remote_url")
        if not url:
            raise SystemExit(
                f"{name} has no public archive URL yet. Upload the release archive to "
                "Zenodo/Dataverse and set remote_url in data/data_registry.json."
            )
        destination = REPO / package["install_dir"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"digital_ca1_{name}_") as temporary:
            archive = Path(temporary) / "package.tar.gz"
            print(f"[DOWNLOAD] {name}: {url}")
            urllib.request.urlretrieve(url, archive)
            unpacked = Path(temporary) / "unpacked"
            unpacked.mkdir()
            with tarfile.open(archive, "r:gz") as stream:
                stream.extractall(unpacked, filter="data")
            roots = [path for path in unpacked.iterdir()]
            source = roots[0] if len(roots) == 1 and roots[0].is_dir() else unpacked
            if destination.exists():
                raise SystemExit(f"Refusing to overwrite existing {destination}")
            shutil.move(str(source), destination)
            print(f"[INSTALLED] {destination}")

    print("Run: python scripts/verify_full_data.py --checksums")


if __name__ == "__main__":
    main()

