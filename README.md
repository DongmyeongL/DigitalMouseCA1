# DigitalMouseCA1

A geometry-aware, cell-type-resolved, full-cable CA1 network model for linking
cellular current to extracellular LFP/CSD.

## What is connected in this release?

The default workflow is the complete model, not a reduced demonstration:

| Quantity | Full model |
|---|---:|
| Cells | 39,112 |
| CA1 model classes | 12 |
| Electrical compartments | 15,118,188 |
| Recurrent edges | 691,641 |
| Synaptic contacts | 2,877,810 |
| Electrical discretization | 60 µm |
| LFP contacts | 210, curved and flat geometry |
| Forward model | finite line source |

The four runtime packages occupy approximately 13.5 GB. Generated simulation
outputs require additional free space.

## Full-model quick start in the development workspace

The existing verified data can be connected without copying it:

```bash
cd DigitalMouseCA1_public
python3 scripts/link_local_full_data.py
python3 scripts/verify_full_data.py
python3 workflows/run_full_model.py --verify-only
```

To compile and run the canonical 500-ms full simulation:

```bash
python3 workflows/run_full_model.py
```

The workflow compiles `cuda/cellwise_recurrent_synaptic.cu` with `nvcc`, runs
the 39,112-cell Hines--Crank--Nicolson solver, generates compartment-current
finite-line-source LFP, and writes `outputs/full_39112_default/`.

## Public download workflow

After the four archives are deposited in Zenodo or Dataverse, insert their
permanent URLs in `data/data_registry.json`. A new user will then run:

```bash
python3 scripts/download_full_data.py --package all
python3 scripts/verify_full_data.py --checksums
python3 workflows/run_full_model.py
```

The current registry deliberately leaves `remote_url` empty: the full derived
runtime packages have not yet been deposited in a public data repository. This
prevents a user from silently downloading an unversioned or unpublished file.

## Runtime packages

1. `electrical_morphology`: cell-specific 60-µm electrical trees and 19-channel parameters.
2. `recurrent_synapses`: edge physiology and 2,877,810 postsynaptic compartment targets.
3. `schaffer_input`: biologically calibrated compound Schaffer collateral sites.
4. `line_leadfield`: 210-contact finite-line-source matrix and type/layer labels.

All expected filenames and byte counts are recorded in
`data/data_registry.json`. `scripts/verify_full_data.py` refuses to launch the
solver if a required file is missing or has the wrong size.

## Canonical outputs

```text
outputs/full_39112_default/
├── simulation_report.json
├── spikes_time_sorted.bin
├── spike_count_by_neuron_u32.bin
├── sample_soma_voltage_f32.bin
├── lfp_time_relative_recording_ms_f32.bin
├── lfp_uV_f32.bin
├── lfp_by_mtype_uV_f32.bin
├── lfp_by_mtype_layer_uV_f32.bin
├── imem_net_nA_f32.bin
└── imem_absolute_nA_f32.bin
```

The report records the complete run configuration, model dimensions, nonfinite
compartment count, KCL residual, spike count and CUDA runtime.

## Hardware and software

- Linux
- NVIDIA CUDA toolkit with `nvcc`
- A CUDA GPU with sufficient memory for the 12.7-GB lead field plus cable,
  channel, synapse and output buffers
- Python 3.9 or newer for data management and orchestration
- At least 20 GB free storage beyond the downloaded runtime packages

Exact GPU memory and runtime depend on CUDA architecture and recording length;
these must be benchmarked and reported for the public v1.0 release.

## Biological scope

The model combines mouse CA1 geometry and soma/transcriptomic information with
predominantly rat morphology and electrophysiology priors. Transcriptomic and
morphological classes are not one-to-one. Some pathway parameters are direct
measurements and others are declared phenotype generalizations. See
`NOTICE.md` before reuse or redistribution.

## Status

This is a local public-release staging repository (`v0.1.0`). Code and all four
full-model package interfaces are connected. Public data download becomes
operational when versioned archives and their DOI URLs are entered in the data
registry.

