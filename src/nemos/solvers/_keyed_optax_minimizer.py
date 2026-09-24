from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx


class _KeyedOptaxState(eqx.Module):
    """Wraps ``_OptaxState`` with the key for the *next* step."""

    key: jnp.ndarray
    inner: Any


class KeyedOptaxMinimiser(optx.OptaxMinimiser):
    """``OptaxMinimiser`` that rolls a PRNG key living in ``args``, not in ``y``.

    ``key_index`` needs a default because the inherited fields already have some.
    """

    key_index: int = -1

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        inner = super().init(fn, y, args, options, f_struct, aux_struct, tags)
        return _KeyedOptaxState(key=args[self.key_index], inner=inner)

    def step(self, fn, y, args, options, state, tags):
        key, subkey = jax.random.split(state.key)
        args = eqx.tree_at(lambda a: a[self.key_index], args, subkey)
        new_y, inner, aux = super().step(fn, y, args, options, state.inner, tags)
        return new_y, _KeyedOptaxState(key=key, inner=inner), aux

    # needed trivial re-implementations otherwise optimistix raises
    def terminate(self, fn, y, args, options, state, tags):
        return super().terminate(fn, y, args, options, state.inner, tags)

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        return super().postprocess(
            fn, y, aux, args, options, state.inner, tags, result
        )
