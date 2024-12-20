import matplotlib.pyplot as plt
import numpy as np

import nemos as nmo
from nemos._inspect_utils.inspect_utils import trim_kwargs

plt.rcParams.update(
    {
        "figure.dpi": 300,
    }
)

KWARGS = dict(
    n_basis_funcs=10,
    decay_rates=np.arange(1, 10 + 1),
    enforce_decay_to_zero=True,
    order=4,
    width=2,
)


def plot_basis(cls):
    cls_params = cls._get_param_names()
    new_kwargs = trim_kwargs(cls, KWARGS.copy(), {cls.__name__: cls_params})
    bas = cls(**new_kwargs)
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    ax.plot(*bas.evaluate_on_grid(300), lw=4)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)


def plot_raised_cosine_linear():
    plot_basis(nmo.basis.RaisedCosineLinearEval)


def plot_raised_cosine_log():
    plot_basis(nmo.basis.RaisedCosineLogEval)


def plot_mspline():
    plot_basis(nmo.basis.MSplineEval)


def plot_bspline():
    plot_basis(nmo.basis.BSplineEval)


def plot_cyclic_bspline():
    plot_basis(nmo.basis.CyclicBSplineEval)


def plot_orth_exp_basis():
    plot_basis(nmo.basis.OrthExponentialEval)


def plot_nd_basis_thumbnail():
    a_basis = nmo.basis.MSplineEval(n_basis_funcs=15, order=3)
    b_basis = nmo.basis.RaisedCosineLogEval(n_basis_funcs=14)
    prod_basis = a_basis * b_basis

    x_coord = np.linspace(0, 1, 1000)
    y_coord = np.linspace(0, 1, 1000)

    X, Y, Z = prod_basis.evaluate_on_grid(200, 200)

    # basis element pairs
    element_pairs = [[0, 0], [5, 1], [10, 5]]

    # plot the 1D basis element and their product
    fig, axs = plt.subplots(3, 3, figsize=(8, 6))
    cc = 0
    for i, j in element_pairs:
        # plot the element form a_basis
        axs[cc, 0].plot(x_coord, a_basis.compute_features(x_coord), "grey", alpha=0.3)
        axs[cc, 0].plot(x_coord, a_basis.compute_features(x_coord)[:, i], "b")
        axs[cc, 0].set_title(f"$a_{{{i}}}(x)$", color="b")

        # plot the element form b_basis
        axs[cc, 1].plot(y_coord, b_basis.compute_features(y_coord), "grey", alpha=0.3)
        axs[cc, 1].plot(y_coord, b_basis.compute_features(y_coord)[:, j], "b")
        axs[cc, 1].set_title(f"$b_{{{j}}}(y)$", color="b")

        # select & plot the corresponding product basis element
        k = i * b_basis.n_basis_funcs + j
        axs[cc, 2].contourf(X, Y, Z[:, :, k], cmap="Blues")
        axs[cc, 2].set_title(
            rf"$A_{{{k}}}(x,y) = a_{{{i}}}(x) \cdot b_{{{j}}}(y)$", color="b"
        )
        axs[cc, 2].set_xlabel("x-coord")
        axs[cc, 2].set_ylabel("y-coord")
        axs[cc, 2].set_aspect("equal")

        cc += 1
    axs[2, 0].set_xlabel("x-coord")
    axs[2, 1].set_xlabel("y-coord")

    plt.tight_layout()


def plot_1d_basis_thumbnail():
    order = 4
    n_basis = 10
    bspline = nmo.basis.BSplineEval(n_basis_funcs=n_basis, order=order)
    x, y = bspline.evaluate_on_grid(100)

    plt.figure(figsize=(5, 3))
    plt.plot(x, y, lw=2)
    plt.title("B-Spline Basis")


def plot_identity_basis():
    plt.figure()
    ax = plt.subplot(111, aspect="equal")
    plt.plot([-1, 1], [-1, 1], lw=6, color="tomato", label="f(x) = x")
    ax.axhline(0, -1, 1, color="k")
    ax.axvline(0, -1, 1, color="k")
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.legend(fontsize=30)
    plt.show()


def plot_history_basis():
    n_basis = KWARGS["n_basis_funcs"]
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    for i in np.linspace(0, 1, n_basis):
        plt.plot([i, i], [0, 1], lw=4)
    for side in ["right", "top"]:
        ax.spines[side].set_visible(False)
    plt.gca().spines["bottom"].set_linewidth(2)
    plt.gca().spines["left"].set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.ylim(0, 1.2)
    plt.xlabel("lag", fontsize=20)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.15)
