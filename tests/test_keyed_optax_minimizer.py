"""Tests for ``KeyedOptaxMinimiser``.

The solver's whole job is to roll a PRNG key that lives in ``args`` instead of in
``y``: every step must split the key it holds and hand the fresh subkey to the
objective. These tests pin that contract down, plus the state (un)wrapping that
makes it work inside the optimistix loop.

``key_index`` is left at its default throughout: the inherited ``__init__`` takes
no such argument, so the key is always the last entry of ``args``.
"""

import jax
import jax.numpy as jnp
import optax
import optimistix as optx
import pytest

from nemos.solvers._keyed_optax_minimizer import KeyedOptaxMinimiser, _KeyedOptaxState

pytestmark = pytest.mark.solver_related

OPTIONS = {}
TAGS = frozenset()

# both PRNG key flavours: old-style uint32[2] and new-style typed keys
KEY_FACTORIES = [jax.random.PRNGKey, jax.random.key]


def _key_bits(key):
    """Raw uint32 data of ``key``, for old-style and typed PRNG keys alike."""
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(key)
    return key


def _same_key(key1, key2):
    """True if the two keys carry the same bits."""
    return bool(jnp.array_equal(_key_bits(key1), _key_bits(key2)))


class KeySpy:
    """A quadratic objective that records the ``args`` it is called with.

    ``OptaxMinimiser.step`` evaluates the objective more than once per step (value
    and gradient), so ``keys`` collapses consecutive duplicates: it lists the
    distinct keys in order of first appearance, which is one per step if the
    solver rolls the key and a single entry if it does not.
    """

    def __init__(self):
        self.args_seen = []

    def __call__(self, y, args):
        self.args_seen.append(args)
        return jnp.sum((y - 3.0) ** 2), None

    @property
    def keys(self):
        """The distinct keys the objective saw, in order."""
        seen = []
        for args in self.args_seen:
            key = args[-1]
            if not seen or not _same_key(seen[-1], key):
                seen.append(key)
        return seen


def _make_args(key):
    """``args`` holding ``key`` last, plus one array that must not move."""
    return (jnp.arange(3.0), key)


def _init(solver, fn, y, args):
    """Run ``solver.init``.

    The structs are spelled out rather than taken from ``jax.eval_shape``: the
    objectives here are scalar with no aux, and tracing them would hand the spy
    tracers that escape the trace.
    """
    f_struct = jax.ShapeDtypeStruct((), jnp.asarray(y).dtype)
    return solver.init(fn, y, args, OPTIONS, f_struct, None, TAGS)


def _expected_chain(key, n_steps):
    """The subkeys ``n_steps`` splits hand out, and the key left over afterwards."""
    subkeys = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        subkeys.append(subkey)
    return subkeys, key


@pytest.fixture
def solver():
    return KeyedOptaxMinimiser(optax.sgd(0.1), rtol=1e-6, atol=1e-6)


class TestState:
    """The keyed state wraps the optax state and is unwrapped on the way back in."""

    def test_init_wraps_the_optax_state(self, solver):
        """``init`` returns a keyed state holding an unmodified optax state."""
        fn = KeySpy()
        y = jnp.zeros(2)
        args = _make_args(jax.random.PRNGKey(0))
        plain = optx.OptaxMinimiser(solver.optim, rtol=1e-6, atol=1e-6)

        state = _init(solver, fn, y, args)

        assert isinstance(state, _KeyedOptaxState)
        assert type(state.inner) is type(_init(plain, fn, y, args))

    @pytest.mark.parametrize("key_factory", KEY_FACTORIES)
    def test_init_picks_the_key_out_of_args(self, solver, key_factory):
        """``init`` seeds the state with the key sitting in ``args``."""
        key = key_factory(0)

        state = _init(solver, KeySpy(), jnp.zeros(2), _make_args(key))

        assert _same_key(state.key, key)

    def test_terminate_accepts_the_keyed_state(self, solver):
        """``terminate`` unwraps the state instead of reading ``.terminate`` off it."""
        fn = KeySpy()
        y = jnp.zeros(2)
        args = _make_args(jax.random.PRNGKey(0))
        state = _init(solver, fn, y, args)

        terminate, result = solver.terminate(fn, y, args, OPTIONS, state, TAGS)

        assert terminate.dtype == jnp.bool_
        assert result == optx.RESULTS.successful

    def test_postprocess_accepts_the_keyed_state(self, solver):
        """``postprocess`` unwraps the state too, and leaves the solution alone."""
        fn = KeySpy()
        y = jnp.zeros(2)
        args = _make_args(jax.random.PRNGKey(0))
        state = _init(solver, fn, y, args)

        out_y, out_aux, stats = solver.postprocess(
            fn, y, None, args, OPTIONS, state, TAGS, optx.RESULTS.successful
        )

        assert jnp.array_equal(out_y, y)
        assert out_aux is None
        assert stats == {}


class TestKeySplitting:
    """Each step splits the key it holds and passes the subkey on."""

    @pytest.mark.parametrize("key_factory", KEY_FACTORIES)
    def test_step_passes_the_subkey(self, solver, key_factory):
        """The objective sees ``split(key)[1]``; the state keeps ``split(key)[0]``."""
        fn = KeySpy()
        y = jnp.zeros(2)
        key = key_factory(0)
        args = _make_args(key)
        expected_key, expected_subkey = jax.random.split(key)

        state = _init(solver, fn, y, args)
        _, state, _ = solver.step(fn, y, args, OPTIONS, state, TAGS)

        assert len(fn.keys) == 1
        assert _same_key(fn.keys[0], expected_subkey)
        assert not _same_key(fn.keys[0], key)
        assert _same_key(state.key, expected_key)

    @pytest.mark.parametrize("key_factory", KEY_FACTORIES)
    def test_consecutive_steps_follow_the_split_chain(self, solver, key_factory):
        """Four steps hand out four different subkeys, in split order."""
        n_steps = 4
        fn = KeySpy()
        y = jnp.zeros(2)
        key = key_factory(0)
        args = _make_args(key)
        expected_subkeys, expected_key = _expected_chain(key, n_steps)

        state = _init(solver, fn, y, args)
        for _ in range(n_steps):
            y, state, _ = solver.step(fn, y, args, OPTIONS, state, TAGS)

        assert len(fn.keys) == n_steps
        assert all(
            _same_key(seen, expected)
            for seen, expected in zip(fn.keys, expected_subkeys)
        )
        assert _same_key(state.key, expected_key)

    def test_only_the_key_moves(self, solver):
        """``args`` is rebuilt around the new key; every other entry is untouched."""
        fn = KeySpy()
        y = jnp.zeros(2)
        args = _make_args(jax.random.PRNGKey(0))

        state = _init(solver, fn, y, args)
        for _ in range(2):
            y, state, _ = solver.step(fn, y, args, OPTIONS, state, TAGS)

        assert len(fn.args_seen) > 0
        assert all(len(seen) == len(args) for seen in fn.args_seen)
        assert all(jnp.array_equal(seen[0], args[0]) for seen in fn.args_seen)

    def test_the_caller_key_is_not_mutated(self, solver):
        """Stepping does not touch the key the caller passed in ``args``."""
        fn = KeySpy()
        y = jnp.zeros(2)
        key = jax.random.PRNGKey(0)
        args = _make_args(key)

        state = _init(solver, fn, y, args)
        solver.step(fn, y, args, OPTIONS, state, TAGS)

        assert _same_key(args[-1], key)


def _noisy_quadratic(y, args):
    """Quadratic whose gradient is perturbed by the key, so the key stream shows up."""
    _, key = args
    return jnp.sum((y - 3.0) ** 2) + jnp.sum(y * jax.random.normal(key, y.shape)), None


def _minimise(solver, fn, key, max_steps=100):
    return optx.minimise(
        fn,
        solver,
        jnp.zeros(3),
        args=_make_args(key),
        has_aux=True,
        max_steps=max_steps,
        throw=False,
    ).value


class TestMinimise:
    """The solver still drives a full ``optx.minimise``, key stream included."""

    def test_deterministic_objective_converges(self, solver):
        """A key-independent problem is solved as usual: the loop runs end to end."""
        sol = optx.minimise(
            KeySpy(),
            solver,
            jnp.zeros(2),
            args=_make_args(jax.random.PRNGKey(0)),
            has_aux=True,
            max_steps=200,
        )

        assert sol.result == optx.RESULTS.successful
        assert jnp.allclose(sol.value, 3.0, atol=1e-3)

    def test_same_key_same_solution(self, solver):
        """The key stream is a pure function of the key passed in ``args``."""
        first = _minimise(solver, _noisy_quadratic, jax.random.PRNGKey(0))
        second = _minimise(solver, _noisy_quadratic, jax.random.PRNGKey(0))

        assert jnp.array_equal(first, second)

    def test_different_key_different_solution(self, solver):
        """The key actually reaches the objective inside the loop."""
        first = _minimise(solver, _noisy_quadratic, jax.random.PRNGKey(0))
        second = _minimise(solver, _noisy_quadratic, jax.random.PRNGKey(1))

        assert not jnp.array_equal(first, second)

    def test_key_rolls_during_the_solve(self, solver):
        """A plain ``OptaxMinimiser`` reuses one key; this one does not."""
        plain = optx.OptaxMinimiser(solver.optim, rtol=solver.rtol, atol=solver.atol)

        keyed = _minimise(solver, _noisy_quadratic, jax.random.PRNGKey(0))
        fixed = _minimise(plain, _noisy_quadratic, jax.random.PRNGKey(0))

        assert not jnp.allclose(keyed, fixed)
