from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..typing import Params, Pytree


def _asarray(dtype, x):
    return jnp.asarray(x, dtype=dtype)


# Work around JAX issue #15676
_asarray = jax.custom_jvp(_asarray, nondiff_argnums=(0,))


@_asarray.defjvp
def _asarray_jvp(dtype, x, tx):
    (x,) = x
    (tx,) = tx
    return _asarray(dtype, x), _asarray(dtype, tx)


def asarray(x):
    dtype = jnp.result_type(x)
    return _asarray(dtype, x)


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def inexact_asarray(x):
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        dtype = default_floating_dtype()
    return _asarray(dtype, x)


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
    # adapted from optimistix._misc
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
