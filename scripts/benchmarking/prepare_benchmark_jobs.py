"""Prepare a disBatch task file for GLM benchmarking on a Slurm cluster.

This script is specific for the Flatiron slurm cluster and makes use of disBatch.

https://github.com/flatironinstitute/disBatch

Usage
-----
# 1. Generate configs and write the disBatch task file:
#    python scripts/prepare_benchmark_jobs.py \\
#        --venv /path/to/venv/bin/activate \\
#        --base_dir /path/to/output/dir

# 2. Copy-paste the printed commands to launch. Each device job ID is captured
#    and passed as a --dependency to a third aggregation job, which runs only
#    after all device jobs succeed.

# To re-run aggregation manually:
#    JAX_PLATFORMS=cpu python scripts/prepare_benchmark_jobs.py --aggregate \\
#        --output_path <base_dir>/results --csv_path <web_output_dir>/<run>.csv
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from benchmarking_glm import (
    DEFAULT_DEVICES,
    DEFAULT_FEATURE_DIMS,
    DEFAULT_N_REPS,
    DEFAULT_PACKAGES,
    DEFAULT_POP_SIZES,
    DEFAULT_REGULARIZERS,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_SOLVER_NAMES,
    _setup,
    aggregate_results,
    combine_summary_statistics,
    generate_all_data,
    generate_glm_configs,
    run_benchmarks,
)

_SELF = Path(__file__)


def _print_env_info() -> None:
    """Print environment diagnostics for debugging worker configuration."""
    import os
    import socket
    import sys

    import jax

    import nemos as nmo

    print("=" * 60)
    print("WORKER ENV DIAGNOSTICS")
    print("=" * 60)
    print(f"  hostname       : {socket.gethostname()}")
    print(f"  python         : {sys.executable}")
    print(f"  python version : {sys.version.split()[0]}")
    print(f"  nemos version  : {nmo.__version__}")
    print(f"  jax version    : {jax.__version__}")
    print(f"  JAX_PLATFORMS  : {os.environ.get('JAX_PLATFORMS', '(not set)')}")
    try:
        devices = jax.devices()
        print(f"  jax devices    : {devices}")
    except Exception as e:
        print(f"  jax devices    : ERROR — {e}")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    cuda_in_path = any(
        "cuda" in p.lower() or "cudnn" in p.lower() for p in ld.split(":")
    )
    print(f"  CUDA in LD_LIBRARY_PATH: {cuda_in_path}")
    if not cuda_in_path:
        print(f"  LD_LIBRARY_PATH: {ld or '(not set)'}")
    print("=" * 60)


def generate_configs(args) -> list:
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    configs = generate_glm_configs(
        sample_sizes=args.sample_sizes,
        feature_dims=args.feature_dims,
        population_sizes=args.pop_sizes,
        regularizers=args.regularizers,
        solver_names=args.solver_names,
        devices=args.devices,
        packages=args.packages,
    )
    config_path = base_dir / "configs.json"
    config_path.write_text(json.dumps(configs, indent=2))
    print(f"Generated {len(configs)} configs -> {config_path}")
    return configs


def write_disbatch_script(args, device: str, indices: list[int]) -> Tuple[Path, int]:
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
                f"source {args.cuda_env}" if device == "gpu" else "true",
                f"source {args.venv}",
                # Expose pip-installed nvidia-* .so files to the dynamic linker.
                # Required for jax-cuda12-plugin to find cuSPARSE, cuDNN, etc.
                # Only needed on GPU workers; harmless (empty path) on CPU.
                "export LD_LIBRARY_PATH=$(python -c \"import os,site; d=site.getsitepackages()[0]+'/nvidia'; print(':'.join(os.path.join(d,p,'lib') for p in os.listdir(d) if os.path.isdir(os.path.join(d,p,'lib')))) if os.path.isdir(d) else ''\" 2>/dev/null):${LD_LIBRARY_PATH:-}",
                # If you set platform to 'gpu' JAX to try all GPU backends
                # and it tries ROCm first, fails with GpuAllocatorConfig.
                # If you set cuda explicitly, then it works fine.
                f"export JAX_PLATFORMS={'cuda' if device == 'gpu' else device}",
                # NEMOS_DATA_DIR: tells fetch_data() where to find cached NWB files.
                # XDG_CACHE_HOME: pynwb tries to create ~/.cache at import time;
                # worker nodes may not have /home/<user>, only /mnt/home/<user>.
                *(
                    [f"export NEMOS_DATA_DIR={args.nemos_data_dir}"]
                    if args.nemos_data_dir
                    else []
                ),
                f"export XDG_CACHE_HOME={Path(args.base_dir) / 'pynwb_cache'}",
                (
                    f"python -u {_SELF}"
                    f" --run"
                    f" --config_path {base_dir / 'configs.json'}"
                    f" --fit_ids {fit_ids_str}"
                    f" --output_path {base_dir / 'results'}"
                    f" --data_path {base_dir / 'data'}"
                    f" --n_reps {args.n_reps}"
                ),
            ]
            f.write(f'( {" && ".join(lines)} ) &> {log}\n')

        f.write("#DISBATCH BARRIER\n")

    n_tasks = len(batches)
    print(
        f"  {dsb_path}  ({len(indices)} configs, {n_tasks} tasks, {args.fits_per_worker} fits/task)"
    )
    return dsb_path, n_tasks


def _build_combine_sbatch_command(args, agg_job_id: str) -> str:
    """Return sbatch command for the summary-statistics combine job.

    Runs only after the aggregation job succeeds (--dependency=afterok).
    """
    combine_log = Path(args.base_dir) / "logs" / "combine.log"
    combine_cmd = (
        f"source {args.venv} && "
        f"export JAX_PLATFORMS=cpu && "
        f"python -u {_SELF}"
        f" --combine"
        f" --csv_path {args.csv_path}"
    )
    return (
        f"sbatch"
        f" --dependency=afterok:{agg_job_id}"
        f" --kill-on-invalid-dep=yes"
        f" -p {args.cpu_partition}"
        f" -t 0-01:00"
        f" --mem-per-cpu=4GB"
        f" -c 1"
        f" -o {combine_log}"
        f" --wrap='{combine_cmd}'"
    )


def _build_aggregation_sbatch_command(args, job_ids: list[str]) -> str:
    """Return sbatch command for the cross-device aggregation job.

    Runs only after all device jobs succeed (--dependency=afterok).
    Always forces JAX to CPU so the import doesn't try to init CUDA.
    """
    base_dir = Path(args.base_dir)
    agg_log = base_dir / "logs" / "aggregate.log"
    dependency = "afterok:" + ":".join(job_ids)
    agg_cmd = (
        f"source {args.venv} && "
        f"export JAX_PLATFORMS=cpu && "
        f"python -u {_SELF}"
        f" --aggregate"
        f" --output_path {base_dir / 'results'}"
        f" --csv_path {args.csv_path}"
    )
    return (
        f"sbatch"
        f" --dependency={dependency}"
        f" --kill-on-invalid-dep=yes"
        f" -p {args.cpu_partition}"
        f" -t 0-05:00"
        f" --mem-per-cpu=4GB"
        f" -c 1"
        f" -o {agg_log}"
        f" --wrap='{agg_cmd}'"
    )


def _build_sbatch_command(args, device: str, dsb_path: Path, n_tasks: int) -> str:
    """Return the full shell command to submit one device's disBatch job."""
    base_dir = Path(args.base_dir)
    partition_for = {"cpu": args.cpu_partition, "gpu": args.gpu_partition}
    disbatch_logs = base_dir / "logs" / device / "disbatch"
    n_workers = min(n_tasks, args.max_workers)
    sbatch_flags = (
        f"-n {n_workers}"
        f" -p {partition_for[device]}"
        f" -t {args.time}"
        f" --mem-per-cpu={args.mem_per_cpu}"
        f" -c {args.cpus_per_task}"
        f" -o {disbatch_logs}/slurm-%j.out"
    )
    if device == "gpu":
        sbatch_flags += f" --gpus-per-task={args.gpus_per_task}"
    return (
        f"module load disBatch && "
        f"mkdir -p {disbatch_logs} && "
        f"sbatch {sbatch_flags} disBatch -p {disbatch_logs}/ {dsb_path}"
    )


def print_commands(args, dsb_paths: dict[str, Path], n_tasks: dict[str, int]) -> None:
    print("\nTo launch (capture job IDs, then submit dependent aggregation):")
    var_names = {}
    for device, dsb_path in dsb_paths.items():
        cmd = _build_sbatch_command(args, device, dsb_path, n_tasks[device])
        var = f"{device.upper()}_JID"
        var_names[device] = var
        print(f"\n  # {device.upper()}")
        print(f"  {var}=$({cmd} | awk '{{print $NF}}')")
    agg_cmd = _build_aggregation_sbatch_command(
        args, [f"${v}" for v in var_names.values()]
    )
    print(f"\n  # Aggregation (runs after all device jobs succeed)")
    print(f"  AGG_JID=$({agg_cmd} | awk '{{print $NF}}')")
    combine_cmd = _build_combine_sbatch_command(args, "$AGG_JID")
    print(f"\n  # Summary statistics (runs after aggregation succeeds)")
    print(f"  {combine_cmd}")


def submit_jobs(args, dsb_paths: dict[str, Path], n_tasks: dict[str, int]) -> None:
    """Submit one sbatch job per device, then a dependent aggregation job."""
    job_ids = []
    for device, dsb_path in dsb_paths.items():
        cmd = _build_sbatch_command(args, device, dsb_path, n_tasks[device])
        print(f"\nSubmitting {device.upper()} jobs:\n  {cmd}")
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        job_id = result.stdout.strip().split()[-1]
        print(f"  -> job ID: {job_id}")
        job_ids.append(job_id)

    agg_cmd = _build_aggregation_sbatch_command(args, job_ids)
    print(f"\nSubmitting aggregation job (depends on {job_ids}):\n  {agg_cmd}")
    result = subprocess.run(agg_cmd, shell=True, check=True, capture_output=True, text=True)
    agg_job_id = result.stdout.strip().split()[-1]
    print(f"  -> job ID: {agg_job_id}")

    combine_cmd = _build_combine_sbatch_command(args, agg_job_id)
    print(f"\nSubmitting summary job (depends on {agg_job_id}):\n  {combine_cmd}")
    subprocess.run(combine_cmd, shell=True, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate GLM benchmarking jobs, or act as a worker entry point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # worker modes (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run",
        action="store_true",
        help="Worker mode: run benchmark fits for --fit_ids.",
    )
    mode.add_argument(
        "--aggregate",
        action="store_true",
        help="Worker mode: aggregate JSON results in --output_path into --csv_path.",
    )
    mode.add_argument(
        "--combine",
        action="store_true",
        help="Worker mode: merge all tagged-commit CSVs in --csv_path's directory into aggregate_summary.csv.",
    )

    # worker: run args
    parser.add_argument(
        "--config_path",
        type=str,
        default="glm_benchmark_configs.json",
        help="Path to the config list JSON (--run mode).",
    )
    parser.add_argument(
        "--fit_ids",
        type=int,
        nargs="+",
        default=None,
        help="Config indices to benchmark. Defaults to all (--run mode).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="benchmark_data",
        help="Directory for cached synthetic data files (--run mode).",
    )

    # shared worker arg
    parser.add_argument(
        "--output_path",
        type=str,
        default="benchmark_results",
        help="Directory for per-config JSON result files.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Aggregated CSV output path (--aggregate mode or orchestration).",
    )
    parser.add_argument("--n_reps", type=int, default=DEFAULT_N_REPS)

    # orchestration-only cluster paths
    parser.add_argument(
        "--venv", default=None, help="Path to the venv activate script."
    )
    parser.add_argument(
        "--cuda_env",
        default=None,
        help="Path to shell script that loads CUDA/cuDNN modules (sourced on GPU workers). Required when devices includes 'gpu'.",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Root directory for configs, results, data, and logs.",
    )
    parser.add_argument(
        "--nemos_data_dir",
        default=None,
        help="Directory where NWB files are cached (sets NEMOS_DATA_DIR on workers). "
        "Required when benchmarking real data.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs via sbatch after writing the disBatch scripts. "
        "Without this flag, only prints the commands (dry-run).",
    )

    # job batching
    parser.add_argument(
        "--fits_per_worker",
        type=int,
        default=1,
        help="Number of configs processed sequentially per disBatch task.",
    )
    parser.add_argument("--cpu_partition", type=str, default="gen")
    parser.add_argument("--gpu_partition", type=str, default="gpu")
    parser.add_argument("--time", type=str, default="0-20:00")
    parser.add_argument("--mem_per_cpu", type=str, default="8GB")
    parser.add_argument("--cpus_per_task", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument(
        "--gpus_per_task", type=int, default=1, help="GPUs per task for GPU jobs."
    )

    # grid parameters
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
    parser.add_argument("--packages", type=str, nargs="+", default=DEFAULT_PACKAGES)

    args = parser.parse_args()

    # validate orchestration-required args when not in worker mode
    if not args.run and not args.aggregate and not args.combine:
        missing = [
            f
            for f, v in [
                ("--venv", args.venv),
                ("--base_dir", args.base_dir),
                ("--csv_path", args.csv_path),
            ]
            if v is None
        ]
        if missing:
            parser.error(f"orchestration mode requires: {', '.join(missing)}")

    return args


def generate_data(args, configs: list) -> None:
    """Pre-generate all unique synthetic datasets so workers only read."""
    base_dir = Path(args.base_dir)
    generate_all_data(configs, str(base_dir / "data"))


if __name__ == "__main__":
    _setup()
    args = _parse_args()

    if args.run:
        _print_env_info()
        configs = json.loads(Path(args.config_path).read_text())
        fit_ids = (
            args.fit_ids if args.fit_ids is not None else list(range(len(configs)))
        )
        run_benchmarks(
            configs, fit_ids, args.output_path, args.data_path, n_reps=args.n_reps
        )

    elif args.aggregate:
        aggregate_results(args.output_path, args.csv_path)

    elif args.combine:
        combine_summary_statistics(args.csv_path)

    else:
        if "gpu" in args.devices and args.cuda_env is None:
            raise SystemExit("--cuda_env is required when devices includes 'gpu'")
        configs = generate_configs(args)

        print("\nPre-generating synthetic datasets ...")
        generate_data(args, configs)

        indices_by_device: dict[str, list[int]] = {}
        for idx, cfg in enumerate(configs):
            indices_by_device.setdefault(cfg["device"], []).append(idx)

        print(f"\nWriting disBatch scripts ({len(configs)} configs total):")
        dsbatch_out = [
            write_disbatch_script(args, device, indices)
            for device, indices in indices_by_device.items()
        ]
        dsb_paths = {
            device: path
            for device, (path, _) in zip(indices_by_device, dsbatch_out, strict=True)
        }
        n_tasks = {
            device: n_task
            for device, (_, n_task) in zip(indices_by_device, dsbatch_out, strict=True)
        }
        print_commands(args, dsb_paths, n_tasks)
        if args.submit:
            submit_jobs(args, dsb_paths, n_tasks)
