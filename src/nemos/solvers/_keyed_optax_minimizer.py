from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from optimistix import OptaxMinimiser

_use_keyed_solver = contextvars.ContextVar("_use_keyed_solver", default=False)


def use_keyed_solver() -> bool:
    """Return True when inside a :func:`keyed_solver` context."""
    return _use_keyed_solver.get()


@contextmanager
def keyed_solver(enabled: bool = True):
    """Make the solver adapters constructed in this context use their keyed minimiser.

    An adapter reads the flag once, in ``__init__``, and keeps its choice afterwards,
    so ``run`` and ``update`` may be called outside the context. An adapter that has
    no keyed minimiser raises at construction.

    Token-based set/reset makes it exception-safe, correctly nested, and isolated
    per thread / async context. Wrap only the single solver construction, never a
    whole method body, so incidental constructions are not accidentally keyed.

    Parameters
    ----------
    enabled :
        When False, this is a no-op and the plain minimiser is used. Lets a call
        site request the keyed minimiser conditionally.
    """
    if not enabled:
        yield
        return
    token = _use_keyed_solver.set(True)
    try:
        yield
    finally:
        _use_keyed_solver.reset(token)


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
    solver_state: Any  # will contain the state of the solver we are wrapping


class KeyedOptaxMinimiser(optx.OptaxMinimiser):
    _key_index: ClassVar[int] = -1

    @_append_optimistix_doc(optx.OptaxMinimiser.init)
    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        """Initialise the inner state and store the key read from ``args[_key_index]``."""
        solver_state = super().init(fn, y, args, options, f_struct, aux_struct, tags)
        return _KeyedOptaxState(key=args[self._key_index], solver_state=solver_state)

    @_append_optimistix_doc(optx.OptaxMinimiser.step)
    def step(self, fn, y, args, options, state, tags):
        """Update the random key and perform one step of the iterative solver."""
        key, subkey = jax.random.split(state.key)
        args = eqx.tree_at(lambda a: a[self._key_index], args, subkey)
        new_y, solver_state, aux = super().step(
            fn, y, args, options, state.solver_state, tags
        )
        return new_y, _KeyedOptaxState(key=key, solver_state=solver_state), aux

    # needed trivial re-implementations otherwise optimistix raises
    @_append_optimistix_doc(optx.OptaxMinimiser.terminate)
    def terminate(self, fn, y, args, options, state, tags):
        """Check termination of iterative solve."""
        return super().terminate(fn, y, args, options, state.solver_state, tags)

    @_append_optimistix_doc(optx.OptaxMinimiser.postprocess)
    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        """Collect solver statistics."""
        return super().postprocess(
            fn, y, aux, args, options, state.solver_state, tags, result
        )


KeyedOptaxMinimiser.__init__.__doc__ = (
    """Wrap ``OptaxMinimiser`` to draw a fresh PRNG key at every step.

At each ``step`` the key held in the state is split in two. One half is written as
the last element of ``args``, and is passed to ``fn`` in that position. The other half
is stored in the new state and will be used in the following step. ``init``
seeds the state with the key it reads from the last element of ``args``, so ``args``
has to be an indexable pytree whose last element is a key.

The position is fixed by ``_key_index`` and is not configurable: this class
inherits ``optx.OptaxMinimiser.__init__``, which does not take it.

The intended use case is a compiled optimization loop requiring independent random
draws, for example a loss that estimates an integral by Monte Carlo and needs fresh
samples at each iteration.

"""
    + OptaxMinimiser.__init__.__doc__
)
