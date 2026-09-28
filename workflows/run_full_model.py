#!/usr/bin/env python3
"""Compile and run the canonical 39,112-cell full-cable CUDA simulation."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "full_39112.json")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    run(["python3", "scripts/verify_full_data.py"])
    if args.verify_only:
        return
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc was not found. Install a CUDA toolkit or use --verify-only.")

    executable = REPO / "build" / "cellwise_recurrent_synaptic"
    executable.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_compile:
        run([
            "nvcc", "-O3", "--use_fast_math", "-std=c++14",
            "cuda/cellwise_recurrent_synaptic.cu", "-o", executable,
        ])
    elif not executable.is_file():
        raise SystemExit(f"Missing compiled solver: {executable}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = REPO / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    external = "none"
    command = [
        executable,
        REPO / "data/full/electrical_morphology",
        REPO / "data/full/recurrent_synapses",
        output,
        config["duration_ms"], config["dt_ms"], config["warmup_ms"],
        config["synaptic_scale"], config["ou_noise_fraction"], config["ou_tau_ms"],
        config["ou_mean_low"], config["ou_mean_high"], config["release_mode"],
        external,
        REPO / "data/full/schaffer_input",
        config["compound_mode"], config["compound_rate_hz"], config["theta_hz"],
        config["compound_scale"], REPO / "data/full/line_leadfield",
        config["lfp_dt_ms"],
    ]
    run(command)
    report = output / "simulation_report.json"
    if not report.is_file():
        raise SystemExit(f"Simulation finished without {report}")
    result = json.loads(report.read_text(encoding="utf-8"))
    print("\nSIMULATION COMPLETE")
    for key in ("cells", "compartments", "edges", "synapses", "spikes",
                "active_cells", "nonfinite_compartments_final",
                "max_global_KCL_relative_residual", "cuda_kernel_ms"):
        print(f"  {key}: {result.get(key)}")


if __name__ == "__main__":
    main()

