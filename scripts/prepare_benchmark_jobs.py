"""Prepare a disBatch task file for GLM benchmarking on a Slurm cluster.

Usage
-----
# 1. Generate configs and write the disBatch task file:
#    python scripts/prepare_benchmark_jobs.py \\
#        --venv /path/to/venv/bin/activate \\
#        --base_dir /path/to/output/dir

# 2. Copy-paste the printed sbatch command to launch.

# After all jobs finish, aggregate:
#    python scripts/benchmarking_glm.py --aggregate \\
#        --output_path <base_dir>/results --csv_path <base_dir>/benchmark_results.csv
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Import grid defaults from the benchmarking script
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from benchmarking_glm import (
    DEFAULT_DEVICES,
    DEFAULT_FEATURE_DIMS,
    DEFAULT_N_REPS,
    DEFAULT_POP_SIZES,
    DEFAULT_REGULARIZERS,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_SOLVER_NAMES,
)

BENCHMARKING_SCRIPT = _HERE / "benchmarking_glm.py"


def generate_configs(args) -> list:
    base_dir = Path(args.base_dir)
    cmd = [
        sys.executable,
        str(BENCHMARKING_SCRIPT),
        "--generate_configs",
        "--config_path",
        str(base_dir / "configs.json"),
        "--sample_sizes",
        *[str(s) for s in args.sample_sizes],
        "--feature_dims",
        *[str(f) for f in args.feature_dims],
        "--pop_sizes",
        *[str(p) for p in args.pop_sizes],
        "--regularizers",
        *args.regularizers,
        "--solver_names",
        *args.solver_names,
        "--devices",
        *args.devices,
    ]
    subprocess.run(cmd, check=True)
    return json.loads((base_dir / "configs.json").read_text())


def write_disbatch_script(args, device: str, indices: list[int]) -> Path:
    """Write one disBatch task file for a single device."""
    base_dir = Path(args.base_dir)
    log_dir = base_dir / "logs" / device
    log_dir.mkdir(exist_ok=True, parents=True)
    (base_dir / "results").mkdir(exist_ok=True, parents=True)
    (base_dir / "data").mkdir(exist_ok=True, parents=True)

    dsb_path = base_dir / f"benchmark_{device}.dsb"
    batches = [
        indices[i : i + args.fits_per_worker]
        for i in range(0, len(indices), args.fits_per_worker)
    ]

    with open(dsb_path, "w") as f:
        for batch in batches:
            fit_ids_str = " ".join(str(i) for i in batch)
            log = log_dir / f"benchmark_{batch[0]:04d}-{batch[-1]:04d}.log"
            lines = [
                f"source {args.venv}",
                (
                    f"python -u {BENCHMARKING_SCRIPT}"
                    f" --config_path {base_dir / 'configs.json'}"
                    f" --fit_ids {fit_ids_str}"
                    f" --output_path {base_dir / 'results'}"
                    f" --data_path {base_dir / 'data'}"
                    f" --n_reps {args.n_reps}"
                ),
            ]
            f.write(f'( {" && ".join(lines)} ) &> {log}\n')

        # barrier within this device's job — no cross-device aggregation here
        f.write("#DISBATCH BARRIER\n")

    n_tasks = len(batches)
    print(
        f"  {dsb_path}  ({len(indices)} configs, {n_tasks} tasks, {args.fits_per_worker} fits/task)"
    )
    return dsb_path


def print_commands(args, dsb_paths: dict[str, Path]) -> None:
    base_dir = Path(args.base_dir)
    partition_for = {"cpu": args.cpu_partition, "gpu": args.gpu_partition}

    print("\nTo launch (one sbatch per device):")
    for device, dsb_path in dsb_paths.items():
        disbatch_logs = base_dir / "logs" / device / "disbatch"
        partition = partition_for[device]
        sbatch_flags = (
            f"-n {args.n_workers}"
            f" -p {partition}"
            f" -t {args.time}"
            f" --mem-per-cpu={args.mem_per_cpu}"
            f" -c {args.cpus_per_task}"
        )
        if device == "gpu":
            sbatch_flags += f" --gpus-per-task={args.gpus_per_task}"
        print(
            f"\n  # {device.upper()}\n"
            f"  module load disBatch; "
            f"mkdir -p {disbatch_logs}; "
            f"sbatch {sbatch_flags} disBatch -p {disbatch_logs}/ {dsb_path}"
        )

    print("\nAfter ALL device jobs finish, aggregate:")
    print(
        f"  source {args.venv} && "
        f"python {BENCHMARKING_SCRIPT}"
        f" --aggregate"
        f" --output_path {base_dir / 'results'}"
        f" --csv_path {base_dir / 'benchmark_results.csv'}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare disBatch task file for GLM benchmarking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required cluster paths
    parser.add_argument(
        "--venv", required=True, help="Path to the venv activate script."
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Root directory for configs, results, data, and logs.",
    )

    # job batching
    parser.add_argument(
        "--fits_per_worker",
        type=int,
        default=1,
        help="Number of configs processed sequentially per disBatch task.",
    )

    # Slurm settings
    parser.add_argument(
        "--n_workers",
        type=int,
        default=10,
        help="Number of disBatch workers (sbatch -n).",
    )
    parser.add_argument("--cpu_partition", type=str, default="cpu")
    parser.add_argument("--gpu_partition", type=str, default="gpu")
    parser.add_argument("--time", type=str, default="0-04:00")
    parser.add_argument("--mem_per_cpu", type=str, default="16GB")
    parser.add_argument("--cpus_per_task", type=int, default=4)
    parser.add_argument(
        "--gpus_per_task", type=int, default=1, help="GPUs per task for GPU jobs."
    )

    # grid parameters — same defaults as benchmarking_glm.py
    parser.add_argument(
        "--sample_sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES
    )
    parser.add_argument(
        "--feature_dims", type=int, nargs="+", default=DEFAULT_FEATURE_DIMS
    )
    parser.add_argument("--pop_sizes", type=int, nargs="+", default=DEFAULT_POP_SIZES)
    parser.add_argument(
        "--regularizers", type=str, nargs="+", default=DEFAULT_REGULARIZERS
    )
    parser.add_argument(
        "--solver_names", type=str, nargs="+", default=DEFAULT_SOLVER_NAMES
    )
    parser.add_argument("--devices", type=str, nargs="+", default=DEFAULT_DEVICES)
    parser.add_argument("--n_reps", type=int, default=DEFAULT_N_REPS)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    configs = generate_configs(args)

    # group config indices by device — one dsb file per device
    indices_by_device: dict[str, list[int]] = {}
    for idx, cfg in enumerate(configs):
        indices_by_device.setdefault(cfg["device"], []).append(idx)

    print(f"\nWriting disBatch scripts ({len(configs)} configs total):")
    dsb_paths = {
        device: write_disbatch_script(args, device, indices)
        for device, indices in indices_by_device.items()
    }
    print_commands(args, dsb_paths)
