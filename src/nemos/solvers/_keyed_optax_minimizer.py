from __future__ import annotations

import inspect
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx


def _append_optimistix_doc(parent_method):
    """Append ``parent_method``'s optimistix docstring below the override's own."""

    def decorator(method):
        header = (
            f"Below, the docstring of `optimistix`'s `{parent_method.__qualname__}`."
        )
        parent_doc = (
            inspect.getdoc(parent_method) or "No documentation found in optimistix."
        )
        method.__doc__ = "\n\n".join(
            (inspect.cleandoc(method.__doc__), header, parent_doc)
        )
        return method

    return decorator


class _KeyedOptaxState(eqx.Module):
    """Wraps ``_OptaxState`` with the key for the *next* step."""

    key: jnp.ndarray
    inner: Any


class KeyedOptaxMinimiser(optx.OptaxMinimiser):
    """``OptaxMinimiser`` that rolls a PRNG key living in ``args``, not in ``y``.

    ``key_index`` needs a default because the inherited fields already have some.
    """

    key_index: int = -1

    @_append_optimistix_doc(optx.OptaxMinimiser.init)
    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        """Initialise the inner state and store the key read from ``args[key_index]``."""
        inner = super().init(fn, y, args, options, f_struct, aux_struct, tags)
        return _KeyedOptaxState(key=args[self.key_index], inner=inner)

    @_append_optimistix_doc(optx.OptaxMinimiser.step)
    def step(self, fn, y, args, options, state, tags):
        """Update the random key and perform one step of the iterative solver."""
        key, subkey = jax.random.split(state.key)
        args = eqx.tree_at(lambda a: a[self.key_index], args, subkey)
        new_y, inner, aux = super().step(fn, y, args, options, state.inner, tags)
        return new_y, _KeyedOptaxState(key=key, inner=inner), aux

    # needed trivial re-implementations otherwise optimistix raises
    @_append_optimistix_doc(optx.OptaxMinimiser.terminate)
    def terminate(self, fn, y, args, options, state, tags):
        """Check termination of iterative solve."""
        return super().terminate(fn, y, args, options, state.inner, tags)

    @_append_optimistix_doc(optx.OptaxMinimiser.postprocess)
    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        """Collect solver statistics."""
        return super().postprocess(fn, y, aux, args, options, state.inner, tags, result)
