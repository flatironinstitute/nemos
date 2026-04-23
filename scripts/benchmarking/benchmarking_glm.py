"""Core benchmarking logic for GLM solvers.

This module is not invoked directly. Use prepare_benchmark_jobs.py to generate
configs, write disBatch task files, and submit jobs. Workers call this module
via prepare_benchmark_jobs.py --run.
"""

import datetime
import hashlib
import json
import socket
import subprocess
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import pandas as pd
import pynapple as nap
from scipy_adapter import ScipyLBFGS
from sklearn.linear_model import PoissonRegressor

import nemos as nmo


def _setup() -> None:
    """Configure JAX and register custom solvers. Must be called before any fitting."""
    nmo.solvers.register("LBFGS", ScipyLBFGS, "scipy")


# --- grid defaults ---
DEFAULT_SAMPLE_SIZES = [100, 1_000, 10_000, 100_000]
DEFAULT_FEATURE_DIMS = [1, 10, 100]
DEFAULT_POP_SIZES = [1, 10, 20]
DEFAULT_REGULARIZERS = ["Ridge", "Lasso"]
DEFAULT_SOLVER_NAMES = [
    # smooth — benchmarked against Ridge; filtered out for Lasso via allowed_reg
    "LBFGS[optax+optimistix]",
    "LBFGS[scipy]",
    "BFGS[optimistix]",
    "GradientDescent[optimistix]",
    "SVRG[nemos]",
    # non-smooth — benchmarked against Lasso only (filtered below)
    "ProximalGradient[optimistix]",
    "ProxSVRG[nemos]",
]
DEFAULT_PACKAGES = ["nemos", "sklearn"]
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
    packages: Optional[Literal["nemos", "sklearn"]] = None,
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
    packages:
        The packages to benchmark, options are "nemos"  or "sklearn".

    Returns
    -------
    The serializable configs.

    """
    if packages is None:
        packages = DEFAULT_PACKAGES
    allowed_reg = {
        name: getattr(nmo.regularizer, name)._allowed_solvers
        for name in dir(nmo.regularizer)
        if not name.startswith("_")
        and hasattr(getattr(nmo.regularizer, name), "_allowed_solvers")
    }
    # Proximal solvers are restricted to non-smooth regularizers. Although NeMoS
    # technically allows ProximalGradient/ProxSVRG.
    _prox_solvers = frozenset({"ProximalGradient", "ProxSVRG"})
    _svrg_solvers = frozenset({"SVRG", "ProxSVRG"})
    _smooth_regs = frozenset({"Ridge", "UnRegularized"})

    configs = []
    for samp, feat, pop_size, reg, dev in product(
        sample_sizes, feature_dims, population_sizes, regularizers, devices
    ):
        input_shapes = {
            "X": [samp, feat],
            "y": [samp, pop_size] if pop_size > 1 else [samp],
        }

        if "nemos" in packages:
            for solv in solver_names:
                base_name = solv.split("[")[0]
                if reg not in allowed_reg or base_name not in allowed_reg[reg]:
                    continue
                if base_name in _prox_solvers and reg in _smooth_regs:
                    continue
                solver_kw = {"maxiter": 1000, "tol": 1e-6}
                if base_name in _svrg_solvers:
                    solver_kw["batch_size"] = max(1, samp // 10)
                fit_config = {
                    "package": "nemos",
                    "input_shapes": input_shapes,
                    "model_conf": {
                        "solver_name": solv,
                        "solver_kwargs": solver_kw,
                        "regularizer": reg,
                        "regularizer_strength": (
                            0.001 if reg != "UnRegularized" else None
                        ),
                    },
                    "device": dev,
                }
                fit_config["file_name"] = f"{dev}_{dict_to_filename(fit_config)}.json"
                configs.append(fit_config)

        if "sklearn" in packages and reg == "Ridge" and dev == "cpu":
            fit_config = {
                "package": "sklearn",
                "input_shapes": input_shapes,
                "model_conf": {
                    "alpha": 0.001,
                    "max_iter": 1000,
                    "solver": "newton-cholesky",
                },
                "device": "cpu",
            }
            fit_config["file_name"] = f"cpu_{dict_to_filename(fit_config)}.json"
            configs.append(fit_config)

    # add HD dataset
    path = nmo.fetch.fetch_data("Mouse32-140822.nwb")
    kwargs = {
        "rate_threshold": 1,
        "epoch_tag": "wake",
        "location": "adn",
        "n_basis_funcs": 8,
        "bin_size": 0.01,
        "window_size": 80,
    }
    X, y = get_hd_data(path, **kwargs)
    input_shapes = {"X": [*X.shape], "y": [*y.shape]}
    extra = {"get_hd_data_kwargs": kwargs, "file_name": path}
    for reg, dev in product(regularizers, devices):
        if "nemos" in packages:
            for solv in solver_names:
                base_name = solv.split("[")[0]
                if reg not in allowed_reg or base_name not in allowed_reg[reg]:
                    continue
                if base_name in _prox_solvers and reg in _smooth_regs:
                    continue
                solver_kw = {"maxiter": 1000, "tol": 1e-6}
                if base_name in _svrg_solvers:
                    solver_kw["batch_size"] = max(1, X.shape[0] // 10)
                configs.append(
                    {
                        "package": "nemos",
                        "input_shapes": input_shapes,
                        "model_conf": {
                            "regularizer": reg,
                            "solver_name": solv,
                            "solver_kwargs": solver_kw,
                            "regularizer_strength": 0.001,
                        },
                        "device": dev,
                        **extra,
                    }
                )

        if "sklearn" in packages and reg == "Ridge" and dev == "cpu":
            configs.append(
                {
                    "package": "sklearn",
                    "input_shapes": input_shapes,
                    "model_conf": {
                        "alpha": 0.001,
                        "max_iter": 1000,
                        "solver": "newton-cholesky",
                    },
                    "device": "cpu",
                    **extra,
                }
            )
    return configs


def _sklearn_model(config) -> list[PoissonRegressor]:
    y_shape = config["input_shapes"]["y"]
    n_neurons = 1 if len(y_shape) == 1 else y_shape[1]
    return [PoissonRegressor(**config["model_conf"]) for _ in range(n_neurons)]


def model_from_config(
    config: dict,
) -> nmo.glm.GLM | nmo.glm.PopulationGLM | list[PoissonRegressor]:
    if config["package"] == "sklearn":
        return _sklearn_model(config)
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
    X = jax.random.normal(keys[0], shape=(samp, feat), dtype=jnp.float64)
    w = 0.1 * jax.random.normal(keys[1], shape=(feat, n_neurons), dtype=jnp.float64)
    weights = w[:, 0] if n_neurons == 1 else w
    intercept = -0.1 * jnp.ones(n_neurons, dtype=jnp.float64)
    if not isinstance(model, list):
        mdl = model.__sklearn_clone__()
    else:
        # sklearn
        cls = nmo.glm.GLM if n_neurons == 1 else nmo.glm.PopulationGLM
        mdl = cls()
    mdl.coef_ = weights
    mdl.intercept_ = intercept
    y, _ = mdl.simulate(keys[2], X)
    return X, y


def get_hd_data(
    path,
    rate_threshold=1.0,
    bin_size=0.01,
    n_basis_funcs=5,
    window_size=80,
    epoch_tag="wake",
    location="adn",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    path = Path(path)
    if not path.exists():
        path = nmo.fetch.fetch_data(path.name)
    data = nap.load_file(path)
    spikes = data["units"]
    epochs = data["epochs"]
    wake_ep = epochs[epochs.tags == epoch_tag]
    spikes = spikes.getby_category("location")[location]
    spikes = spikes.restrict(wake_ep).getby_threshold("rate", rate_threshold)
    y = spikes.count(bin_size, ep=wake_ep)
    X = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs, window_size=window_size
    ).compute_features(y)
    X, y = jnp.asarray(X.d, dtype=jnp.float64), jnp.asarray(y.d, dtype=jnp.float64)
    keep = jnp.all(~jnp.isnan(X), axis=1)
    return X[keep], y[keep]


def generate_all_data(configs: list, data_path: str) -> None:
    """Pre-generate all unique synthetic datasets and save to data_path.

    Called once before workers are submitted so that fitting workers only read,
    avoiding parallel write races. Real-data (NWB) configs are skipped.
    """
    root = Path(data_path)
    root.mkdir(exist_ok=True, parents=True)
    seen = set()
    for config in configs:
        if config["file_name"].endswith("nwb"):
            continue
        key = dict_to_filename(config["input_shapes"])
        if key in seen:
            continue
        seen.add(key)
        file_path = root / (key + ".npz")
        if file_path.exists():
            print(f"  exists: {file_path.name}")
            continue
        model = model_from_config(config)
        X, y = generate_data(model, config)
        jnp.savez(str(file_path), X=X, y=y)
        print(f"  saved:  {file_path.name}  X={X.shape} y={y.shape}")
    print(f"Data generation complete ({len(seen)} unique shapes).")


def get_data(config: dict, path: str = ".") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load data for a config. Synthetic data is read from pre-generated npz files."""
    if config["file_name"].endswith("nwb"):
        return get_hd_data(config["file_name"], **config["get_hd_data_kwargs"])
    file_path = Path(path) / (dict_to_filename(config["input_shapes"]) + ".npz")
    npz = jnp.load(str(file_path))
    return jnp.asarray(npz["X"], dtype=jnp.float64), jnp.asarray(npz["y"], dtype=jnp.float64)


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).parent),
                    "rev-parse",
                    "--short",
                    "HEAD",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def _get_all_tagged_commits() -> dict[str, str]:
    lst = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).parent),
                "for-each-ref",
                "--format='%(objectname:short) %(refname:short)'",
                "refs/tags",
            ]
        )
        .decode()
        .strip()
        .replace("'", "")
        .split("\n")
    )
    return dict(s.split(" ") for s in lst if s)


def _get_cpu_model() -> str:
    try:
        out = subprocess.check_output(["lscpu"], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return "unknown"


def _get_gpu_models() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _get_meta() -> dict:
    return {
        "nemos_version": nmo.__version__,
        "jax_version": jax.__version__,
        "hostname": socket.gethostname(),
        "cpu_model": _get_cpu_model(),
        "gpu_models": _get_gpu_models(),
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": _get_git_commit(),
    }


def _benchmark_nemos(config: dict, X: jnp.ndarray, y: jnp.ndarray, n_reps: int) -> dict:
    """Benchmark a NeMoS GLM/PopulationGLM, isolating solver init, compilation, and execution."""
    model = model_from_config(config)
    pars = model.initialize_params(X, y)
    model_pars = model._validator.to_model_params(pars)

    is_scipy = model.solver_spec.backend == "scipy"

    def _get_iter_num(m):
        if is_scipy:
            return int(m.solver_state_.iter_num)
        return int(m.solver_state_.stats.num_steps.item())

    def _get_converged(state):
        if is_scipy:
            return bool(state.converged)
        return bool(state.stats.converged.item())

    solver_init_s = []
    compilation_s = []
    fit_s = []
    converged = []

    for _ in range(n_reps):
        model = model_from_config(config)

        t0 = perf_counter()
        _ = model._initialize_optimizer_and_state(model_pars, X, y)
        t1 = perf_counter()
        solver_init_s.append(t1 - t0)

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

        t4 = perf_counter()
        pars, state, _ = compiled(model_pars, X, y)
        pars.coef.block_until_ready()
        t5 = perf_counter()
        fit_s.append(t5 - t4)

        converged.append(_get_converged(state))

    # Clear XLA cache so end-to-end reps see a cold cache, independent of the
    # compilation measurements above. Global side effect: invalidates all cached
    # JIT compilations in the current process.
    jax.clear_caches()

    end_to_end_s = []
    num_solver_iter = []
    param_norm = []

    for _ in range(n_reps):
        model = model_from_config(config)
        t6 = perf_counter()
        model.fit(X, y)
        t7 = perf_counter()
        end_to_end_s.append(t7 - t6)
        num_solver_iter.append(_get_iter_num(model))
        param_norm.append(float(jnp.linalg.norm(model.coef_)))

    step_time = [
        f / n if n > 0 else float("nan") for f, n in zip(fit_s, num_solver_iter)
    ]

    input_shapes = config["input_shapes"]
    model_conf = config["model_conf"]
    flat_config = {
        "package": "nemos",
        "sample_size": input_shapes["X"][0],
        "feature_dim": input_shapes["X"][1],
        "pop_size": input_shapes["y"][1] if len(input_shapes["y"]) > 1 else 1,
        "solver_name": model_conf["solver_name"],
        "regularizer": model_conf["regularizer"],
        "maxiter": model_conf["solver_kwargs"]["maxiter"],
        "tol": model_conf["solver_kwargs"]["tol"],
        "batch_size": model_conf["solver_kwargs"].get("batch_size"),
        "device": config["device"],
        "solver_class": model._solver.__class__.__name__,
        "data_source": (
            Path(config["file_name"]).name
            if config["file_name"].endswith("nwb")
            else "synthetic"
        ),
    }

    return {
        "config": flat_config,
        "results": {
            "solver_init_s": solver_init_s,
            "compilation_s": compilation_s,
            "fit_s": fit_s,
            "step_time": step_time,
            "end_to_end_s": end_to_end_s,
            "converged": converged,
            "iter_num": num_solver_iter,
            "param_norm": param_norm,
        },
        "meta": _get_meta(),
    }


def _benchmark_sklearn(
    config: dict, X: jnp.ndarray, y: jnp.ndarray, n_reps: int
) -> dict:
    """Benchmark sklearn PoissonRegressor, timing the sequential per-neuron fit loop."""
    n_neurons = 1 if len(y.shape) == 1 else y.shape[1]
    y_cols = [y] if n_neurons == 1 else [y[:, i] for i in range(n_neurons)]
    nan = float("nan")

    end_to_end_s = []
    converged = []
    num_solver_iter = []
    param_norm = []

    for _ in range(n_reps):
        models = _sklearn_model(config)
        t0 = perf_counter()
        for m, yi in zip(models, y_cols):
            m.fit(X, yi)
        t1 = perf_counter()
        end_to_end_s.append(t1 - t0)

        max_iter = models[0].max_iter
        converged.append(all(m.n_iter_ < max_iter for m in models))
        num_solver_iter.append(sum(int(m.n_iter_) for m in models))
        coefs = jnp.concatenate([jnp.ravel(jnp.array(m.coef_)) for m in models])
        param_norm.append(float(jnp.linalg.norm(coefs)))

    input_shapes = config["input_shapes"]
    model_conf = config["model_conf"]
    flat_config = {
        "package": "sklearn",
        "sample_size": input_shapes["X"][0],
        "feature_dim": input_shapes["X"][1],
        "pop_size": input_shapes["y"][1] if len(input_shapes["y"]) > 1 else 1,
        "solver_name": "newton-cholesky[sklearn]",
        "regularizer": "Ridge",
        "maxiter": model_conf.get("max_iter", 1000),
        "tol": nan,
        "batch_size": None,
        "device": config["device"],
        "solver_class": "PoissonRegressor",
        "data_source": (
            Path(config["file_name"]).name
            if config["file_name"].endswith("nwb")
            else "synthetic"
        ),
    }

    return {
        "config": flat_config,
        "results": {
            "solver_init_s": [nan] * n_reps,
            "compilation_s": [nan] * n_reps,
            "fit_s": [nan] * n_reps,
            "step_time": [nan] * n_reps,
            "end_to_end_s": end_to_end_s,
            "converged": converged,
            "iter_num": num_solver_iter,
            "param_norm": param_norm,
        },
        "meta": _get_meta(),
    }


def benchmark_fit(
    config: dict, X: jnp.ndarray, y: jnp.ndarray, n_reps: int = DEFAULT_N_REPS
) -> dict:
    """Benchmark model fitting for a single config.

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
    if config["package"] == "sklearn":
        return _benchmark_sklearn(config, X, y, n_reps)
    return _benchmark_nemos(config, X, y, n_reps)


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
        is_real_data = config["file_name"].endswith("nwb")
        pkg = config["package"]
        shapes = config["input_shapes"]
        if pkg == "sklearn":
            solver_label = f"newton-cholesky[sklearn] | Ridge | alpha={config['model_conf']['alpha']}"
        else:
            mc = config["model_conf"]
            solver_label = f"{mc['solver_name']} | {mc['regularizer']}"
        data_label = Path(config["file_name"]).name if is_real_data else "Simulation"
        print(
            f"{data_label}: [{idx}] {solver_label} | "
            f"X={shapes['X']} | y={shapes['y']} | device={config['device']}"
        )
        X, y = get_data(config, path=data_path)
        result = benchmark_fit(config, X, y, n_reps=n_reps)
        if is_real_data:
            json_name = Path(config["file_name"]).with_suffix(".json").name
            _hash = dict_to_filename(config)
            out_file = out_dir / f"{_hash}_{json_name}"
        else:
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
                    "step_time": res["step_time"][i],
                    "end_to_end_s": res["end_to_end_s"][i],
                    "converged": res["converged"][i],
                    "iter_num": res["iter_num"][i],
                    "param_norm": res["param_norm"][i],
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


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    # compute the fraction of the end-to-end fit time spent on compilation
    df.loc[:, "compile_time_fraction"] = df["compilation_s"] / (
        df["solver_init_s"] + df["compilation_s"] + df["fit_s"]
    )
    # replace nans with 0s (this is for timed compilation with nan in the index
    df["compile_time_fraction"] = df["compile_time_fraction"].fillna(0)
    summary = (
        df.groupby(
            [
                "version",
                "git_commit",
                "data_source",
                "device",
                "solver_name",
                "sample_size",
                "feature_dim",
                "pop_size",
            ]
        )
        .agg(
            fit_time_s=("end_to_end_s", "mean"),
            converged=("converged", "all"),
            iter_num=("iter_num", "mean"),
            compile_time_fraction=("compile_time_fraction", "mean"),
        )
        .sort_values("fit_time_s")
        .reset_index()
    )
    return summary


def combine_summary_statistics(
    csv_path: str,
):
    csv_dir = Path(csv_path).parent
    tagged_commits = _get_all_tagged_commits()
    # derive current commit from the filename — robust regardless of cwd or checkout
    current_commit = Path(csv_path).stem.split("_")[-1]
    # add latest(or replace tag with latest)
    tagged_commits[current_commit] = "latest"

    # loop over available benchmarking results and aggregate
    dfs = []
    for f in csv_dir.glob("*.csv"):
        commit = f.stem.split("_")[-1]
        # skip non-tagged non-current commits
        if commit not in tagged_commits:
            continue
        try:
            df = pd.read_csv(f)
            commit = df["git_commit"].iloc[0]
            df["version"] = tagged_commits.get(commit, commit)
            dfs.append(df)
        except Exception as e:
            raise RuntimeError(
                "Failed to aggregate benchmark results. Dataframe entries may be incompatible."
            ) from e

    if not dfs:
        print("No matching benchmark CSVs found.")
        return pd.DataFrame()
    aggregate_df = pd.concat(dfs, ignore_index=True)

    summary = compute_summary_stats(aggregate_df)
    summary.to_csv(csv_dir / "aggregate_summary.csv", index=False)
    print(f"Aggregated summary statistics saved to {csv_dir}")
    return summary
