import matplotlib.pyplot as plt
import numpy as np

import nemos as nmo
from nemos._inspect_utils import trim_kwargs

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
    fig, ax = plt.subplots(1, 1, figsize=(5 / 4, 2.5 / 4))
    ax.plot(*bas.evaluate_on_grid(300), lw=0.8)
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
