import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap

import jax
import jax.numpy as jnp

import nemos as nmo
from sklearn.linear_model import PoissonRegressor

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------


def get_data():
    path = nmo.fetch.fetch_data("Mouse32-140822.nwb")
    data = nap.load_file(path)

    spikes = data["units"]
    epochs = data["epochs"]
    wake_ep = epochs[epochs.tags == "wake"]

    spikes = spikes.getby_category("location")["adn"]
    spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.0)

    y = spikes.count(0.01, ep=wake_ep)
    X = nmo.basis.RaisedCosineLogConv(5, window_size=80).compute_features(y)

    X, y = X.d, y.d
    keep = np.all(~np.isnan(X), axis=1)

    return X[keep], y[keep]


# ---------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------

MODES = [
    ("nemos_jit_autodiff", True, False),
    ("nemos_nojit_autodiff", False, False),
    ("nemos_jit_analytic", True, True),
    ("nemos_nojit_analytic", False, True),
]


# ---------------------------------------------------------------------
# RUNNERS
# ---------------------------------------------------------------------


def run_nemos(glm, X, y):
    t0 = time.perf_counter()
    model = glm.fit(X, y)
    t1 = time.perf_counter()
    return model, t1 - t0


def run_sklearn(X, y):
    model = PoissonRegressor(max_iter=1000)

    t0 = time.perf_counter()
    model.fit(X, y)
    t1 = time.perf_counter()

    return model, t1 - t0


def compare(nm, sk):
    return (
        np.linalg.norm(nm.coef_ - sk.coef_),
        np.linalg.norm(np.asarray(nm.intercept_) - np.asarray(sk.intercept_)),
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

X, Y = get_data()
n_neurons = Y.shape[1]

results = []

for neuron in range(n_neurons):
    y = Y[:, neuron]
    X, y = np.array(X), np.array(y)

    print(f"\nNeuron {neuron:02d}")

    # sklearn baseline
    sk_model, t_sk = run_sklearn(X, y)

    for name, jit, analytic in MODES:
        glm = nmo.glm.GLM(
            regularizer="Ridge",
            solver_name="Newton",
            solver_kwargs={
                "jit": jit,
                "force_autodiff_hessian": not analytic,
            },
        )

        model, t = run_nemos(glm, X, y)

        coef_diff, int_diff = compare(model, sk_model)

        results.append(
            {
                "neuron": neuron,
                "method": name,
                "time": t,
                "coef_diff": coef_diff,
                "intercept_diff": int_diff,
            }
        )

    results.append(
        {
            "neuron": neuron,
            "method": "sklearn_cpu",
            "time": t_sk,
            "coef_diff": 0.0,
            "intercept_diff": 0.0,
        }
    )


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

df = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(df.groupby("method")[["time", "coef_diff", "intercept_diff"]].mean())


# ---------------------------------------------------------------------
# PLOTS (with error bars + sorting)
# ---------------------------------------------------------------------


def summarize(metric):
    g = df.groupby("method")[metric]
    mean = g.mean()
    std = g.std()
    return mean, std


def plot_metric(metric, ylabel):
    mean, std = summarize(metric)

    # sort by mean value
    order = mean.sort_values().index
    mean = mean.loc[order]
    std = std.loc[order]

    plt.figure(figsize=(8, 4))

    plt.bar(
        range(len(mean)),
        mean.values,
        yerr=std.values,
        capsize=5,
    )

    plt.xticks(range(len(mean)), mean.index, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()


plot_metric("time", "mean runtime (s)")
plot_metric("coef_diff", "mean coefficient error vs sklearn")
plt.show()
