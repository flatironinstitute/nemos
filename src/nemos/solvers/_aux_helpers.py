"""
Helpers for supporting objective functions with aux.

Relevant NeMoS PR and discussion:
  https://github.com/flatironinstitute/nemos/pull/444
  https://github.com/flatironinstitute/nemos/pull/444#discussion_r2593434556
"""

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

# NOTE: If optimistix._misc._asarray is deleted, just use jnp.asarray instead.
# Requires bumping JAX version: jax>=0.7.2
# That is when the fix for #15676 and representing constants as arrays in jaxpr were included.
#   https://github.com/jax-ml/jax/issues/15676
#   https://docs.jax.dev/en/latest/changelog.html#jax-0-7-2-september-16-2025
# So swtiching to jnp.asarray can also be done if the jax requirement is >=0.7.2.
# Note that arguments are reversed.
from optimistix._misc import _asarray

from ..typing import Params, Pytree


def asarray(x):
    dtype = jnp.result_type(x)  # to prevent weak_type?
    return _asarray(dtype, x)


def inexact_asarray(x):
    """Adaptation of optimistix._misc.inexact_asarray."""
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        if jax.config.jax_enable_x64:  # pyright: ignore
            dtype = jnp.float64
        else:
            dtype = jnp.float32

    return _asarray(dtype, x)


def tree_map_inexact_asarray(params: Pytree):
    """Cast PyTree as an inexact jnp.array."""
    # adapted from optimistix._misc
    return jax.tree_util.tree_map(inexact_asarray, params)


def wrap_aux(fn: Callable) -> Callable:
    """Make function that returns (fn(...), None)."""

    def _fn_with_aux(*args, **kwargs):
        return fn(*args, **kwargs), None

    return _fn_with_aux


def drop_aux(fn: Callable) -> Callable:
    """Make function that returns fn's value without its aux."""

    def _fn_without_aux(*args, **kwargs):
        return fn(*args, **kwargs)[0]

    return _fn_without_aux


def pack_args(fn: Callable) -> Callable:
    """Make function that accepts fn(params, args) instead of fn(params, *args)."""

    def _fn_with_packed_args(params, args):
        return fn(params, *args)

    return _fn_with_packed_args


def out_asarray(fn: Callable):
    """Return output as inexact array, aux as array. Adapted from Optimistix."""

    def _fn_with_array_outputs(*args, **kwargs):
        out, aux = fn(*args, **kwargs)
        out = jax.tree_util.tree_map(inexact_asarray, out)
        aux = jax.tree_util.tree_map(asarray, aux)
        return out, aux

    return _fn_with_array_outputs


def convert_fn(fn: Callable, has_aux: bool, y0: Params, args: Any):
    """Convert the objective function the way optimistix.minimise does."""

    y0 = tree_map_inexact_asarray(y0)
    if not has_aux:
        fn = wrap_aux(fn)
    fn = out_asarray(fn)
    fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore

    return fn
