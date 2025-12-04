from typing import Any, Callable

import equinox as eqx
import jax
from optimistix._misc import asarray, inexact_asarray

from ..typing import Params, Pytree


def _as_inexact_array(params: Pytree):
    """Cast PyTree as an inexact jnp.array."""
    # adapted from optimistix._misc
    return jax.tree_util.tree_map(inexact_asarray, params)


def _wrap_aux(fn: Callable) -> Callable:
    """Make function that returns (fn(...), None)."""

    def _fn_with_aux(*args, **kwargs):
        return fn(*args, **kwargs), None

    return _fn_with_aux


def _drop_aux(fn: Callable) -> Callable:
    """Make function that returns fn's value without its aux."""

    def _fn_without_aux(*args, **kwargs):
        return fn(*args, **kwargs)[0]

    return _fn_without_aux


def _pack_args(fn: Callable) -> Callable:
    """Make function that accepts fn(params, args) instead of fn(params, *args)."""

    def _fn_with_packed_args(params, args):
        return fn(params, *args)

    return _fn_with_packed_args


def _out_asarray(fn: Callable):
    """Return output as inexact array, aux as array. Adapted from Optimistix."""

    def _fn_with_array_outputs(*args, **kwargs):
        out, aux = fn(*args, **kwargs)
        out = jax.tree_util.tree_map(inexact_asarray, out)
        aux = jax.tree_util.tree_map(asarray, aux)
        return out, aux

    return _fn_with_array_outputs


def _convert_fn(fn: Callable, has_aux: bool, y0: Params, args: Any):
    """Convert the objective function the way optimistix.minimise does."""

    y0 = _as_inexact_array(y0)
    if not has_aux:
        fn = _wrap_aux(fn)
    fn = _out_asarray(fn)
    fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore

    return fn
