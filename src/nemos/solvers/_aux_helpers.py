from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..typing import Params, Pytree


def _inexact_asarray(x):
    """
    Cast x as inexact array.

    Adaptation of optimistix._misc.inexact_asarray using
    jnp.asarray instead of optimistix._misc.asarray.
    """
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        if jax.config.jax_enable_x64:  # pyright: ignore
            dtype = jnp.float64
        else:
            dtype = jnp.float32

    # using jnp.asarray instead of optimistix._misc.asarray
    # as the relevant JAX issue seems to be fixed
    # https://github.com/jax-ml/jax/issues/15676
    return jnp.asarray(x, dtype)


def _as_inexact_array(params: Pytree):
    """Cast PyTree as an inexact jnp.array."""
    # adapted from optimistix._misc
    return jax.tree_util.tree_map(_inexact_asarray, params)


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
        out = jax.tree_util.tree_map(_inexact_asarray, out)
        aux = jax.tree_util.tree_map(jnp.asarray, aux)
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
