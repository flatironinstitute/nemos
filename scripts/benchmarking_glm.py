"""Benchmark GLM fitting for different solvers.


Examples
--------
# generate config
python scripts/benchmarking_glm.py \                                                                                                                                     
    --generate_configs \                                                                                                                                                 
    --config_path gpu_configs.json \                                                                                                                                     
    --devices gpu \                                                                                                                                                      
    --feature_dims 1 \                                                                                                                                                   
    --pop_sizes 1

# fit one
python scripts/benchmarking_glm.py \
    --config_path gpu_configs.json \                                                                                                                                     
    --output_path gpu_results \
    --data_path gpu_data \
    --fit_ids 0 \
    --n_reps 1   
"""

import argparse
import datetime
import hashlib
import json
import socket
import subprocess
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import jax
import jax.numpy as jnp
from scipy_adapter import ScipyLBFGS

import nemos as nmo

# register solver
nmo.solvers.register("LBFGS", ScipyLBFGS, "scipy")

# --- grid defaults ---
DEFAULT_SAMPLE_SIZES = [100, 1_000, 10_000, 100_000]
DEFAULT_FEATURE_DIMS = [1, 10, 100]
DEFAULT_POP_SIZES = [1, 10, 20]
DEFAULT_REGULARIZERS = ["UnRegularized", "Ridge"]
DEFAULT_SOLVER_NAMES = [
    "LBFGS[optax+optimistix]",
    "LBFGS[scipy]",
    "GradientDescent[optimistix]",
]
DEFAULT_DEVICES = ["cpu"]
DEFAULT_N_REPS = 10


def dict_to_filename(config: dict, length: int = 16) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def generate_glm_configs(
    sample_sizes: List[int],
    feature_dims: List[int],
    population_sizes: List[int],
    regularizers: List[str],
    solver_names: List[str],
    devices: List[str],
) -> List[dict]:
    """
    Generate GLM configurations to benchmark.

    The configuration will be list of dict. Each dict includes model configuration specs.
    The list is serializable to a json for job parallelization (if needed).

    Parameters
    ----------
    sample_sizes:
        List of the sample sizes to try.
    feature_dims:
        List of the feature dimensions to try.
    population_sizes:
        List of the number of neurons to try. N>1 will fit a PopulationGLM.
    regularizers:
        List of the regularization parameters to try.
    solver_names:
        List of the full solver names to try, i.e. "solvername[backend]".
    devices:
        List of devices to try, e.g. ["cpu", "gpu"].

    Returns
    -------
    The serializable configs.

    """
    allowed_reg = {
        name: getattr(nmo.regularizer, name)._allowed_solvers
        for name in dir(nmo.regularizer)
        if not name.startswith("_")
        and hasattr(getattr(nmo.regularizer, name), "_allowed_solvers")
    }
    configs = []
    for samp, feat, pop_size, reg, solv, dev in product(
        sample_sizes,
        feature_dims,
        population_sizes,
        regularizers,
        solver_names,
        devices,
    ):
        base_name = solv.split("[")[0]
        if reg not in allowed_reg or base_name not in allowed_reg[reg]:
            continue

        fit_config = {
            "input_shapes": {
                "X": [samp, feat],
                "y": [samp, pop_size] if pop_size > 1 else [samp],
            },
            "model_conf": {
                "solver_name": solv,
                "solver_kwargs": {"maxiter": 10000, "tol": 1e-8},
                "regularizer": reg,
            },
            "device": dev,
        }
        # file_name ties each result file back to a unique config
        fit_config["file_name"] = f"{dev}_{dict_to_filename(fit_config)}.json"
        configs.append(fit_config)
    return configs


def model_from_config(config: dict) -> nmo.glm.GLM | nmo.glm.PopulationGLM:
    y_shape = config["input_shapes"]["y"]
    is_population = len(y_shape) > 1
    if is_population:
        return nmo.glm.PopulationGLM(**config["model_conf"])
    return nmo.glm.GLM(**config["model_conf"])


def generate_data(
    model: nmo.glm.PopulationGLM | nmo.glm.GLM, config: dict
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    keys = jax.random.split(jax.random.key(123), 3)
    samp, feat = config["input_shapes"]["X"]
    y_shape = config["input_shapes"]["y"]
    n_neurons = y_shape[1] if len(y_shape) > 1 else 1
    X = jax.random.normal(keys[0], shape=(samp, feat)) / feat
    weights = jnp.squeeze(
        0.1 * jax.random.normal(keys[1], shape=(feat, n_neurons)),
    )
    intercept = -0.1 * jnp.ones(n_neurons)
    model.coef_ = weights
    model.intercept_ = intercept
    y, _ = model.simulate(keys[2], X)
    model.coef_ = None
    model.intercept_ = None
    return X, y


def get_data(
    config: dict, regenerate: bool = False, save: bool = True, path: str = "."
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    root = Path(path)
    root.mkdir(exist_ok=True, parents=True)
    file_path = root / (dict_to_filename(config["input_shapes"]) + ".npz")
    if file_path.exists() and not regenerate:
        npz = jnp.load(str(file_path))
        return jnp.asarray(npz["X"]), jnp.asarray(npz["y"])
    model = model_from_config(config)
    X, y = generate_data(model, config)
    if save:
        jnp.savez(str(file_path), X=X, y=y)
    return X, y


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def benchmark_fit(
    config: dict, X: jnp.ndarray, y: jnp.ndarray, n_reps: int = DEFAULT_N_REPS
) -> dict:
    """Benchmark model fitting, isolating solver init, compilation, and execution.

    Parameters
    ----------
    config:
        A single config dict as produced by generate_glm_configs.
    X:
        Design matrix.
    y:
        Response array.
    n_reps:
        Number of independent repetitions.

    Returns
    -------
    A dict with keys "config", "results", "meta", ready for JSON serialization.
    """
    if config["device"] != X.device.platform or config["device"] != y.device.platform:
        raise RuntimeError(
            "Device configuration doesn't match requested platform. "
            "GPU configs must be run on GPU nodes."
        )

    # initialize_params is deterministic for fixed config/data — compute once
    model = model_from_config(config)
    pars = model.initialize_params(X, y)
    model_pars = model._validator.to_model_params(pars)

    is_scipy = model.solver_spec.backend == "scipy"

    def _get_iter_num(model):
        if is_scipy:
            return int(model.solver_state_.iter_num)
        return int(model.solver_state_.stats.num_steps.item())

    def _get_converged(state):
        if is_scipy:
            return bool(state.converged)
        return bool(state.stats.converged.item())

    solver_init_s = []
    compilation_s = []
    fit_s = []
    end_to_end_s = []
    converged = []
    num_solver_iter = []

    for _ in range(n_reps):
        # fresh model each rep for independent measurements
        model = model_from_config(config)

        t0 = perf_counter()
        _ = model._initialize_optimizer_and_state(model_pars, X, y)
        t1 = perf_counter()
        solver_init_s.append(t1 - t0)

        # compile on the same instance so _optimizer_run is valid
        if not is_scipy:
            t2 = perf_counter()
            compiled = (
                jax.jit(model._optimizer_run).trace(model_pars, X, y).lower().compile()
            )
            t3 = perf_counter()
            compilation_s.append(t3 - t2)
        else:
            compiled = model._optimizer_run
            compilation_s.append(jnp.nan)
        # execution only — no compilation overhead
        t4 = perf_counter()
        pars, state, _ = compiled(model_pars, X, y)
        pars.coef.block_until_ready()
        t5 = perf_counter()
        fit_s.append(t5 - t4)

        has_converged = _get_converged(state)
        converged.append(has_converged)

        # end-to-end fit: includes all preprocessing, validation, param init,
        # solver init, compilation, and execution. Subtracting (solver_init +
        # compilation + fit) from this gives an estimate of preprocessing overhead.
        model = model_from_config(config)
        t6 = perf_counter()
        model.fit(X, y)
        t7 = perf_counter()
        end_to_end_s.append(t7 - t6)
        num_solver_iter.append(_get_iter_num(model))

    input_shapes = config["input_shapes"]
    model_conf = config["model_conf"]
    flat_config = {
        "sample_size": input_shapes["X"][0],
        "feature_dim": input_shapes["X"][1],
        "pop_size": input_shapes["y"][1] if len(input_shapes["y"]) > 1 else 1,
        "solver_name": model_conf["solver_name"],
        "regularizer": model_conf["regularizer"],
        "maxiter": model_conf["solver_kwargs"]["maxiter"],
        "tol": model_conf["solver_kwargs"]["tol"],
        "device": config["device"],
        # ground truth solver used
        "solver_class": model._solver.__class__.__name__,
    }

    return {
        "config": flat_config,
        "results": {
            "solver_init_s": solver_init_s,
            "compilation_s": compilation_s,
            "fit_s": fit_s,
            "end_to_end_s": end_to_end_s,
            "converged": converged,
            "iter_num": num_solver_iter,
        },
        "meta": {
            "nemos_version": nmo.__version__,
            "jax_version": jax.__version__,
            "hostname": socket.gethostname(),
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": _get_git_commit(),
        },
    }


def run_benchmarks(
    configs: List[dict],
    fit_ids: List[int],
    output_path: str,
    data_path: str,
    n_reps: int = DEFAULT_N_REPS,
) -> None:
    """Run benchmarks for selected config indices and save one JSON per config."""
    out_dir = Path(output_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    for idx in fit_ids:
        config = configs[idx]
        mc = config["model_conf"]
        shapes = config["input_shapes"]
        print(
            f"[{idx}] {mc['solver_name']} | {mc['regularizer']} | "
            f"X={shapes['X']} | y={shapes['y']} | device={config['device']}"
        )
        X, y = get_data(config, path=data_path)
        result = benchmark_fit(config, X, y, n_reps=n_reps)
        out_file = out_dir / config["file_name"]
        out_file.write_text(json.dumps(result, indent=2))
        print(f"  -> {out_file}")


def aggregate_results(results_dir: str, csv_path: str) -> None:
    """Aggregate all per-config JSON result files into a single long-format CSV.

    Each row in the CSV corresponds to one repetition of one config, with
    all config fields and metadata included as columns.
    """
    import csv

    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        d = json.loads(f.read_text())
        cfg, res, meta = d["config"], d["results"], d["meta"]
        n_reps = len(res["fit_s"])
        for i in range(n_reps):
            rows.append(
                {
                    **cfg,
                    **meta,
                    "rep": i,
                    "solver_init_s": res["solver_init_s"][i],
                    "compilation_s": res["compilation_s"][i],
                    "fit_s": res["fit_s"][i],
                    "end_to_end_s": res["end_to_end_s"][i],
                    "converged": res["converged"][i],
                    "iter_num": res["iter_num"][i],
                }
            )

    if not rows:
        print(f"No result files found in {results_dir}.")
        return

    n_configs = len(set(f.stem for f in Path(results_dir).glob("*.json")))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {len(rows)} rows ({n_configs} configs) -> {csv_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GLM fitting across solvers and configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # mutually exclusive top-level actions
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--generate_configs",
        action="store_true",
        help="Generate config list from grid parameters and save to --config_path.",
    )
    actions.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate JSON results in --output_path into --csv_path.",
    )

    # paths
    parser.add_argument(
        "--config_path",
        type=str,
        default="glm_benchmark_configs.json",
        help="Path to load/save the config list JSON.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="benchmark_results",
        help="Directory for per-config JSON result files.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="benchmark_data",
        help="Directory for cached synthetic data files.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV path (used with --aggregate).",
    )

    # job selection
    parser.add_argument(
        "--fit_ids",
        type=int,
        nargs="+",
        default=None,
        help="Indices into the config list to benchmark. Defaults to all configs.",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=DEFAULT_N_REPS,
        help="Number of independent repetitions per config.",
    )

    # grid parameters (used with --generate_configs)
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

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.generate_configs:
        configs = generate_glm_configs(
            sample_sizes=args.sample_sizes,
            feature_dims=args.feature_dims,
            population_sizes=args.pop_sizes,
            regularizers=args.regularizers,
            solver_names=args.solver_names,
            devices=args.devices,
        )
        Path(args.config_path).write_text(json.dumps(configs, indent=2))
        print(f"Generated {len(configs)} configs -> {args.config_path}")

    elif args.aggregate:
        aggregate_results(args.output_path, args.csv_path)

    else:
        configs = json.loads(Path(args.config_path).read_text())
        fit_ids = (
            args.fit_ids if args.fit_ids is not None else list(range(len(configs)))
        )
        run_benchmarks(
            configs, fit_ids, args.output_path, args.data_path, n_reps=args.n_reps
        )
