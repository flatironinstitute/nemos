import copy
import re
import warnings
from contextlib import nullcontext as does_not_raise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.linear_model import GammaRegressor, PoissonRegressor

import nemos as nmo
from nemos.glm.params import GLMParams

# Register every test here as solver-related
pytestmark = pytest.mark.solver_related


@pytest.fixture(scope="module", autouse=True)
def register_deregister_agradientdescent():
    """Fixture for registering and deregistering AGradientDescent."""
    name = "AGradientDescent"

    # register a random solver under this name
    nmo.solvers.register(
        name, nmo.solvers.get_solver("LBFGS").implementation, backend="custom"
    )

    yield

    from nemos.solvers._solver_registry import _registry, _defaults

    if name in _registry and "custom" in _registry[name]:
        # remove custom dummy implementation
        _registry[name].pop("custom", None)
        # if there are no other implementations, remove the algo name
        if not _registry[name]:
            _registry.pop(name, None)
    # if the default implementation was the custom dummy one, remove it
    if _defaults.get(name) == "custom":
        _defaults.pop(name, None)


@pytest.mark.parametrize(
    "reg_str, reg_type",
    [
        ("UnRegularized", nmo.regularizer.UnRegularized),
        (None, nmo.regularizer.UnRegularized),
        ("Ridge", nmo.regularizer.Ridge),
        ("Lasso", nmo.regularizer.Lasso),
        ("GroupLasso", nmo.regularizer.GroupLasso),
        ("ElasticNet", nmo.regularizer.ElasticNet),
        ("not_valid", None),
        ("nemos.regularizer.UnRegularized", nmo.regularizer.UnRegularized),
        ("nemos.regularizer.Ridge", nmo.regularizer.Ridge),
        ("nemos.regularizer.Lasso", nmo.regularizer.Lasso),
        ("nemos.regularizer.GroupLasso", nmo.regularizer.GroupLasso),
    ],
)
def test_regularizer_builder(reg_str, reg_type):
    """Test building a regularizer from a string"""
    valid_regularizers = nmo._regularizer_builder.AVAILABLE_REGULARIZERS
    raise_exception = reg_str is not None and not (
        reg_str in valid_regularizers
        or any(reg_str == f"nemos.regularizer.{name}" for name in valid_regularizers)
    )
    if raise_exception:
        with pytest.raises(ValueError, match=f"Unknown regularizer: {reg_str}. "):
            nmo._regularizer_builder.instantiate_regularizer(reg_str)
    else:
        # build a regularizer by string
        regularizer = nmo._regularizer_builder.instantiate_regularizer(reg_str)
        # assert correct type of regularizer is instantiated
        assert isinstance(regularizer, reg_type)
        # create a regularizer of that type
        regularizer2 = reg_type()
        # assert that they have the same attributes
        assert regularizer.__dict__ == regularizer2.__dict__


@pytest.mark.parametrize(
    "expected, reg",
    [
        ("UnRegularized()", nmo.regularizer.UnRegularized()),
        ("Ridge()", nmo.regularizer.Ridge()),
        ("Lasso()", nmo.regularizer.Lasso()),
        ("GroupLasso()", nmo.regularizer.GroupLasso(mask=np.eye(4))),
        ("ElasticNet()", nmo.regularizer.ElasticNet()),
    ],
)
def test_regularizer_repr(reg, expected):
    assert repr(reg) == expected


def test_regularizer_available():
    for regularizer in nmo._regularizer_builder.AVAILABLE_REGULARIZERS:
        reg = nmo._regularizer_builder.instantiate_regularizer(regularizer)
        assert reg.__class__.__name__ == regularizer


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.UnRegularized(),
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso(mask=np.array([[1.0]])),
        nmo.regularizer.ElasticNet(),
    ],
)
def test_get_only_allowed_solvers(regularizer):
    # the error raised by property changed in python 3.11
    with pytest.raises(
        AttributeError,
        match="property 'allowed_solvers' of '.+' object has no setter|can't set attribute",
    ):
        regularizer.allowed_solvers = []


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.UnRegularized(),
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso(mask=np.array([[1.0]])),
        nmo.regularizer.ElasticNet(),
    ],
)
def test_item_assignment_allowed_solvers(regularizer):
    with pytest.raises(
        TypeError, match="'tuple' object does not support item assignment"
    ):
        regularizer.allowed_solvers[0] = "my-favourite-solver"


@pytest.mark.parametrize(
    "regularizer_class",
    [
        nmo.regularizer.UnRegularized,
        nmo.regularizer.Ridge,
        nmo.regularizer.Lasso,
        nmo.regularizer.ElasticNet,
        nmo.regularizer.GroupLasso,
    ],
)
def test_allow_solver(regularizer_class):
    """allow_solver should update the class-level tuple for all instances."""
    new_solver = "MyCoolNewAlgorithm"
    original_allowed = regularizer_class._allowed_solvers

    reg1 = regularizer_class()
    reg2 = regularizer_class()

    # by default it's not allowed
    assert new_solver not in reg1.allowed_solvers
    assert new_solver not in reg2.allowed_solvers

    try:
        # register and allow the this solver
        # using LBFGS just as a dummy that implements the solver interface
        nmo.solvers.register(
            new_solver, nmo.solvers.get_solver("LBFGS").implementation, default=True
        )
        regularizer_class.allow_solver(new_solver)

        assert new_solver in reg1.allowed_solvers
        assert new_solver in reg2.allowed_solvers
        assert new_solver in regularizer_class().allowed_solvers

        with does_not_raise():
            reg1.check_solver(new_solver)
            model = nmo.glm.GLM(regularizer=reg1, solver_name=new_solver)
            assert model.algo_name == new_solver

        # allowing a solver already allowed does nothing
        _default_solver = regularizer_class._default_solver
        regularizer_class.allow_solver(_default_solver)
        assert regularizer_class().allowed_solvers.count(_default_solver) == 1
        assert reg1.allowed_solvers.count(_default_solver) == 1
        assert reg2.allowed_solvers.count(_default_solver) == 1
    finally:
        # reset to avoid leaking the extra solver into other tests
        regularizer_class._allowed_solvers = original_allowed
        nmo.solvers._solver_registry._registry.pop("MyCoolNewAlgorithm")
        nmo.solvers._solver_registry._defaults.pop("MyCoolNewAlgorithm")


@pytest.mark.parametrize(
    "regularizer, regularizer_strength",
    [
        ("Ridge", 2.0),
        ("Lasso", 2.0),
        ("GroupLasso", 2.0),
        ("ElasticNet", (2.0, 0.7)),
    ],
)
def test_set_params_order_change_regularizer(regularizer, regularizer_strength):
    """Test that set_params() when changing regularizer and regularizer_strength regardless of order."""
    # start with unregularized
    model = nmo.glm.GLM()
    assert model.regularizer_strength is None
    assert isinstance(model.regularizer, nmo.regularizer.UnRegularized)

    # set regularizer first
    model.set_params(regularizer=regularizer, regularizer_strength=regularizer_strength)
    assert model.regularizer_strength == regularizer_strength
    assert model.regularizer.__class__.__name__ == regularizer

    # set regularizer_strength first
    model.set_params(regularizer="UnRegularized")
    assert model.regularizer_strength is None

    model.set_params(regularizer_strength=regularizer_strength, regularizer=regularizer)
    assert model.regularizer_strength == regularizer_strength
    assert model.regularizer.__class__.__name__ == regularizer


@pytest.mark.parametrize(
    "regularizer, regularizer_strength",
    [
        ("Ridge", 2.0),
        ("Lasso", 2.0),
        ("GroupLasso", 2.0),
        ("ElasticNet", (2.0, 0.7)),
        ("UnRegularized", None),
    ],
)
@pytest.mark.parametrize(
    "regularizer2, regularizer2_default",
    [
        ("Ridge", 1.0),
        ("Lasso", 1.0),
        ("GroupLasso", 1.0),
        ("ElasticNet", (1.0, 0.5)),
        ("UnRegularized", None),
    ],
)
def test_change_regularizer_reset_strength(
    regularizer,
    regularizer_strength,
    regularizer2,
    regularizer2_default,
):
    """Test that set_params() when changing regularizer and regularizer_strength regardless of order."""
    model = nmo.glm.GLM(
        regularizer=regularizer, regularizer_strength=regularizer_strength
    )
    assert model.regularizer_strength == regularizer_strength
    assert model.regularizer.__class__.__name__ == regularizer

    # check that regularizer_strength is reset when changing regularizer
    model.set_params(regularizer=regularizer2)
    assert model.regularizer_strength == regularizer2_default

    # make sure there is no conflict when setting back
    model.set_params(regularizer=regularizer, regularizer_strength=regularizer_strength)
    assert model.regularizer_strength == regularizer_strength


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso(),
    ],
)
@pytest.mark.parametrize(
    "strength, expectation",
    [
        (None, does_not_raise()),
        (0.5, does_not_raise()),
        (1.0, does_not_raise()),
        (jnp.array(1.0), does_not_raise()),
        (np.array(0.5), does_not_raise()),
        (
            "bah",
            pytest.raises(
                TypeError,
                match=f"Could not convert regularizer strength to floats:",
            ),
        ),
    ],
)
def test_validate_strength_single_input(regularizer, strength, expectation):
    """Test that regularizer accepts scalar strength input (or None)."""
    with expectation:
        result = regularizer._validate_strength(strength)
        if strength is None:
            assert result == 1.0
        else:
            assert result == strength
            assert isinstance(result, float)


@pytest.mark.parametrize(
    "regularizer",
    [nmo.regularizer.Ridge(), nmo.regularizer.Lasso(), nmo.regularizer.GroupLasso()],
)
@pytest.mark.parametrize(
    "strength, expectation, check_fn",
    [
        # Dict with scalar leaves (Python floats)
        (
            {"a": 0.5, "b": 0.3},
            does_not_raise(),
            lambda result: (
                isinstance(result, dict)
                and isinstance(result["a"], float)
                and result["a"] == 0.5
                and isinstance(result["b"], float)
                and result["b"] == 0.3
            ),
        ),
        # Dict with 0-dim arrays (should be converted to float)
        (
            {"a": jnp.array(0.5), "b": np.array(0.3)},
            does_not_raise(),
            lambda result: (
                isinstance(result, dict)
                and isinstance(result["a"], float)
                and result["a"] == 0.5
                and isinstance(result["b"], float)
                and result["b"] == 0.3
            ),
        ),
        # Dict with 1-D arrays (should be preserved as jnp.ndarray)
        (
            {
                "a": jnp.array([0.1, 0.2, 0.3]),
                "b": np.array([0.5]),
            },
            does_not_raise(),
            lambda result: (
                isinstance(result, dict)
                and isinstance(result["a"], jnp.ndarray)
                and result["a"].shape == (3,)
                and jnp.allclose(result["a"], jnp.array([0.1, 0.2, 0.3]))
                and isinstance(result["b"], jnp.ndarray)
                and result["b"].shape == (1,)
                and jnp.allclose(result["b"], jnp.array([0.5]))
            ),
        ),
        # Dict with mixed 0-dim and arrays
        (
            {
                "a": jnp.array(0.5),
                "b": np.array([0.1, 0.2]),
            },
            does_not_raise(),
            lambda result: (
                isinstance(result, dict)
                and isinstance(result["a"], float)
                and result["a"] == 0.5
                and isinstance(result["b"], jnp.ndarray)
                and result["b"].shape == (2,)
            ),
        ),
        # Nested dict (dict of dicts)
        (
            {
                "a": {"x": 0.3, "y": 0.4},
                "b": {"x": jnp.array([0.1, 0.2]), "y": 0.5},
            },
            does_not_raise(),
            lambda result: (
                isinstance(result, dict)
                and isinstance(result["a"]["x"], float)
                and result["a"]["x"] == 0.3
                and isinstance(result["a"]["y"], float)
                and result["a"]["y"] == 0.4
                and isinstance(result["b"]["x"], jnp.ndarray)
                and result["b"]["x"].shape == (2,)
                and isinstance(result["b"]["y"], float)
                and result["b"]["y"] == 0.5
            ),
        ),
        # 1-D array
        (
            jnp.array([0.1, 0.2, 0.3]),
            does_not_raise(),
            lambda result: (
                isinstance(result, jnp.ndarray)
                and result.shape == (3,)
                and jnp.allclose(result, jnp.array([0.1, 0.2, 0.3]))
            ),
        ),
        # 2-D array
        (
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            does_not_raise(),
            lambda result: (
                isinstance(result, jnp.ndarray)
                and result.shape == (2, 2)
                and jnp.allclose(result, jnp.array([[0.1, 0.2], [0.3, 0.4]]))
            ),
        ),
        # Dict with string leaf
        (
            {"a": "invalid", "b": 0.5},
            pytest.raises(
                TypeError, match="Could not convert regularizer strength to floats:"
            ),
            lambda result: True,
        ),
        # Nested dict with invalid leaf
        (
            {
                "a": {"x": 0.3, "y": "bad"},
            },
            pytest.raises(
                TypeError, match="Could not convert regularizer strength to floats:"
            ),
            lambda result: True,
        ),
    ],
)
def test_validate_strength_tree_input(regularizer, strength, expectation, check_fn):
    """Test that regularizer accepts tree strength with proper type conversion.

    Type conversion rules:
    - 0-dim arrays or scalar numbers → float
    - Arrays with shape (n,) or higher → jnp.ndarray with same shape
    - Strings or other invalid types → TypeError
    """
    with expectation:
        result = regularizer._validate_strength(strength)
        assert check_fn(result)


@pytest.mark.parametrize(
    "regularizer",
    [nmo.regularizer.UnRegularized(), nmo.regularizer.Ridge(), nmo.regularizer.Lasso()],
)
@pytest.mark.parametrize("strength", [None, 0.3, np.array(1.0), jnp.array(1.0)])
def test_validate_strength_structure_scalar_broadcast(regularizer, strength):
    """Scalar and 0-d strengths broadcast over regularizable leaves; non-regularizable leaves are None."""
    params = GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0]))

    # Call structure alignment directly with raw strength;
    # base method accepts None/scalars/0-d, doesn't change the type
    structured = regularizer._validate_strength_structure(params, strength)

    assert isinstance(structured, GLMParams)
    # coef gets scalar broadcast (as a scalar per leaf), intercept is None
    expected_scalar = 1.0 if strength is None else float(strength)
    expected_type = float if strength is None else type(strength)
    assert isinstance(structured.coef, expected_type)
    assert structured.coef == expected_scalar
    assert structured.intercept is None


class TwoParams(eqx.Module):
    """Helper module exposing two regularizable subtrees."""

    one: jnp.ndarray | dict
    two: jnp.ndarray | dict

    def regularizable_subtrees(self):
        return [lambda p: p.one, lambda p: p.two]


@pytest.mark.parametrize(
    "regularizer",
    [nmo.regularizer.UnRegularized(), nmo.regularizer.Ridge(), nmo.regularizer.Lasso()],
)
def test_validate_strength_structure_multiple_subtrees_count_mismatch(regularizer):
    """If the number of provided strengths != number of subtrees, raise ValueError."""
    params = TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,)))

    with pytest.raises(ValueError, match=r"Expected 2 strength values, got"):
        regularizer._validate_strength_structure(params, [0.1])  # only one provided


@pytest.mark.parametrize(
    "regularizer",
    [nmo.regularizer.UnRegularized(), nmo.regularizer.Ridge(), nmo.regularizer.Lasso()],
)
@pytest.mark.parametrize(
    "params, strength, expectation",
    [
        # GLMParams: no groups
        (
            GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
            0.1,
            does_not_raise(),
        ),
        (
            GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
            jnp.array([0.1, 0.2, 0.3]),
            does_not_raise(),
        ),
        (
            GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
            jnp.array([0.1, 0.2]),  # not enough
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
            jnp.array([0.1, 0.2, 0.3, 0.4]),  # too many
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        # GLMParams: 2 groups
        (
            GLMParams(
                coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                intercept=jnp.array([0.0]),
            ),
            {"a": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.1, 0.2])},  # match
            does_not_raise(),
        ),
        (
            GLMParams(
                coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                intercept=jnp.array([0.0]),
            ),
            {"a": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.1, 0.2])},  # match
            does_not_raise(),
        ),
        (
            GLMParams(
                coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                intercept=jnp.array([0.0]),
            ),
            {"a": 0.1, "b": 0.1},  # scalar per group
            does_not_raise(),
        ),
        (
            GLMParams(
                coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                intercept=jnp.array([0.0]),
            ),
            {"a": 0.1, "b": jnp.array([0.1, 0.2])},  # mix scalar and array
            does_not_raise(),
        ),
        (
            GLMParams(
                coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                intercept=jnp.array([0.0]),
            ),
            {"a": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.1])},  # not enough
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            GLMParams(
                coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                intercept=jnp.array([0.0]),
            ),
            {"a": 0.1, "b": jnp.array([0.1])},  # mix scalar and not enough
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            GLMParams(
                coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                intercept=jnp.array([0.0]),
            ),
            {
                "a": jnp.array([0.1, 0.2, 0.3, 0.4]),
                "b": 0.1,
            },  # mix scalar and too many
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        # Dictionaries
        (
            {"a": jnp.ones((3,)), "b": jnp.ones((2,))},
            {"a": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.1, 0.2])},
            does_not_raise(),
        ),
        (
            {"a": jnp.ones((3,)), "b": jnp.ones((2,))},
            {"a": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.1])},  # not enough
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            {"a": jnp.ones((3,)), "b": jnp.ones((2,))},
            {
                "a": jnp.array([0.1, 0.2, 0.3, 0.4]),
                "b": jnp.array([0.1, 0.2]),
            },  # too many
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            {"a": jnp.ones((3,)), "b": jnp.ones((2,))},
            {
                "a": 0.1,
                "b": jnp.array([0.1]),
            },  # mix scalar and not enough
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            {"a": jnp.ones((3,)), "b": jnp.ones((2,))},
            {
                "a": jnp.array([0.1, 0.2, 0.3, 0.4]),
                "b": 0.1,
            },  # mix scalar and too many
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        # Arbitrary object with regularizable subtrees: no groups
        (
            TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
            [jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])],
            does_not_raise(),
        ),
        (
            TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
            [0.1, 0.1],
            does_not_raise(),
        ),
        (
            TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
            [0.1, jnp.array([0.1, 0.2])],
            does_not_raise(),
        ),
        (
            TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
            [jnp.array([0.1, 0.2, 0.3]), 0.1],
            does_not_raise(),
        ),
        (
            TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
            [jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])],
            does_not_raise(),
        ),
        (
            TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
            [jnp.array([0.1, 0.2]), jnp.array([0.1, 0.2])],  # not enough
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
            [jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1])],  # not enough
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            TwoParams(
                one=jnp.ones((3,)), two=jnp.ones((2,))
            ),  # mix not enough and scalar
            [jnp.array([0.1, 0.2]), 0.1],
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            TwoParams(
                one=jnp.ones((3,)), two=jnp.ones((2,))
            ),  # mix too many and scalar
            [0.1, jnp.array([0.1, 0.2, 0.3])],
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        # Arbitrary object with regularizable subtrees: mixed groups
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))}, two=jnp.ones((2,))
            ),
            [{"a": 0.1, "b": 0.1}, jnp.array([0.1, 0.2])],
            does_not_raise(),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                two={"c": jnp.ones((5,))},
            ),
            [{"a": 0.1, "b": 0.1}, 0.1],
            does_not_raise(),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                two={"c": jnp.ones((5,))},
            ),
            [{"a": 0.1, "b": 0.1}, {"c": 0.1}],
            does_not_raise(),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                two={"c": jnp.ones((5,))},
            ),
            [{"a": 0.1, "b": 0.1}, {"c": jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])}],
            does_not_raise(),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                two={"c": jnp.ones((5,))},
            ),
            [
                {"a": jnp.array([0.1, 0.2, 0.3]), "b": 0.1},
                {"c": jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])},
            ],
            does_not_raise(),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))}, two=jnp.ones((2,))
            ),
            [],
            pytest.raises(ValueError, match=r"Expected .* strength values, got "),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))}, two=jnp.ones((2,))
            ),
            [{"a": 0.1, "b": 0.1}],
            pytest.raises(ValueError, match=r"Expected .* strength values, got "),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))}, two=jnp.ones((2,))
            ),
            [{"a": 0.1, "b": 0.1}, 0.1, 0.1],
            pytest.raises(ValueError, match=r"Expected .* strength values, got "),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                two={"c": jnp.ones((5,))},
            ),
            [
                {"a": jnp.array([0.1, 0.2, 0.3, 0.4]), "b": 0.1},  # too many
                {"c": jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])},
            ],
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
        (
            TwoParams(
                one={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                two={"c": jnp.ones((5,))},
            ),
            [
                {"a": jnp.array([0.1, 0.2, 0.3]), "b": 0.1},
                {"c": jnp.array([0.1, 0.2, 0.3, 0.4])},  # not enough
            ],
            pytest.raises(
                ValueError, match=r"Strength shape .* does not match parameter shape .*"
            ),
        ),
    ],
)
def test_validate_strength_structure_shape_mismatch(
    regularizer, strength, params, expectation
):
    """Non-scalar array strength must match parameter leaf shape; otherwise raises ValueError."""
    with expectation:
        regularizer._validate_strength_structure(params, strength)


class TestUnRegularized:
    cls = nmo.regularizer.UnRegularized

    def test_filter_kwargs_contains_strength(self):
        """Test that strength is in filter kwargs."""
        n_features = 3
        params = GLMParams(coef=jnp.ones((n_features,)), intercept=jnp.array([0.0]))
        regularizer = self.cls()

        fk = regularizer._get_filter_kwargs(params=params, strength=None)
        assert isinstance(fk, dict)
        assert "strength" in fk

        s = fk["strength"]
        assert isinstance(s, GLMParams)
        assert isinstance(s.coef, float)
        assert s.coef == 1.0
        assert s.intercept is None

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test UnRegularized acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test UnRegularized acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)
        with expectation:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize(
        "strength",
        [
            None,
            0.0,
            1.0,
            jnp.array(1.0),  # 0-d array
            jnp.array([0.1, 0.2]),  # 1-d array
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # 2-d array
            {"a": 0.5, "b": jnp.array([0.1, 0.2])},  # mixed tree
            GLMParams(coef=0.1, intercept=0.5),
            "bah",  # arbitrary string
        ],
    )
    def test_validate_strength(self, strength):
        """UnRegularized ignores strength and always returns None."""
        reg = self.cls()
        assert reg._validate_strength(strength) is None

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize(
        "solver_name", ["GradientDescent", "BFGS", "SVRG", "ProxSVRG"]
    )
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test Ridge acceptable kwargs."""
        regularizer = self.cls()
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
            )

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_is_callable(self, loss):
        """Test Unregularized callable loss."""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # set regularizer and solver name
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)
        model.initialize_optimization_and_state(X, y, init_pars)
        params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        model.optimization_run(params, X, y)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)
        model.initialize_optimization_and_state(X, y, init_pars)
        params = model._validator.to_model_params(
            (
                jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
                true_params.intercept,
            )
        )
        model.optimization_run(
            params,
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_output_match(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model params
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        init_pars = model.initialize_params(X, y)
        model.initialize_optimization_and_state(X, y, init_pars)

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)

        # update solver name
        model_bfgs = copy.deepcopy(model)
        model_bfgs.solver_name = "BFGS"
        init_pars_bfgs = model_bfgs.initialize_params(X, y)
        model_bfgs.initialize_optimization_and_state(X, y, init_pars_bfgs)
        params_gd = model.optimization_run(init_params, X, y)[0]
        params_bfgs = model_bfgs.optimization_run(init_params, X, y)[0]

        match_weights = np.allclose(params_gd.coef, params_bfgs.coef)
        match_intercepts = np.allclose(params_gd.intercept, params_bfgs.intercept)

        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Convex estimators should converge to the same numerical value."
            )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_sklearn(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        init_pars = model.initialize_params(X, y)
        model.initialize_optimization_and_state(X, y, init_pars)
        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        params = model.optimization_run(init_params, X, y)[0]
        model_skl = PoissonRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("UnRegularized GLM estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_sklearn_gamma(
        self, gammaGLM_model_instantiation, solver_name
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.inverse_link_function = jnp.exp
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)
        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        params = model.optimization_run(init_params, X, y)[0]
        model_skl = GammaRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match sklearn!")

    @pytest.mark.parametrize(
        "inv_link_jax, link_sm",
        [
            (jnp.exp, sm.families.links.Log()),
            (lambda x: 1 / x, sm.families.links.InversePower()),
        ],
    )
    @pytest.mark.parametrize("solver_name", ["LBFGS", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_statsmodels_gamma(
        self, inv_link_jax, link_sm, gammaGLM_model_instantiation, solver_name
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.inverse_link_function = inv_link_jax
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-13}
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)
        params = model.optimization_run(
            model._model_specific_initialization(X, y), X, y
        )[0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The InversePower link function does "
            )
            model_sm = sm.GLM(
                endog=y, exog=sm.add_constant(X), family=sm.families.Gamma(link=link_sm)
            )

        res_sm = model_sm.fit(cnvrg_tol=10**-12)

        match_weights = np.allclose(res_sm.params[1:], params.coef)
        match_intercepts = np.allclose(res_sm.params[:1], params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match statsmodels!")

    @pytest.mark.parametrize(
        "inv_link_jax, link_sm",
        [
            (jnp.exp, sm.families.links.Log()),
        ],
    )
    @pytest.mark.parametrize("solver_name", ["LBFGS", "SVRG", "ProximalGradient"])
    @pytest.mark.requires_x64
    def test_solver_match_statsmodels_negative_binomial(
        self,
        inv_link_jax,
        link_sm,
        negativeBinomialGLM_model_instantiation,
        solver_name,
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = negativeBinomialGLM_model_instantiation
        y = y.astype(
            float
        )  # needed since solver.run is called directly, nemos converts.
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.observation_model.inverse_link_function = inv_link_jax
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-13}
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)
        params = model.optimization_run(
            model._model_specific_initialization(X, y), X, y
        )[0]
        model_sm = sm.GLM(
            endog=y,
            exog=sm.add_constant(X),
            family=sm.families.NegativeBinomial(
                link=link_sm, alpha=model.observation_model.scale
            ),
        )

        res_sm = model_sm.fit(cnvrg_tol=10**-12)

        match_weights = np.allclose(res_sm.params[1:], params.coef, atol=10**-6)
        match_intercepts = np.allclose(res_sm.params[:1], params.intercept, atol=10**-6)
        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Unregularized GLM estimate does not match statsmodels!\n"
                f"Intercept difference is: {res_sm.params[:1] - params.intercept}\n"
                f"Coefficient difference is: {res_sm.params[1:] - params.coef}"
            )

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "NonlinearCG",
            "ProximalGradient",
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.fit(X, y)


class TestRidge:
    cls = nmo.regularizer.Ridge

    def test_filter_kwargs_contains_strength(self):
        """Test that strength is in filter kwargs."""
        n_features = 3
        params = GLMParams(coef=jnp.ones((n_features,)), intercept=jnp.array([0.0]))
        regularizer = self.cls()

        fk = regularizer._get_filter_kwargs(params=params, strength=0.7)
        assert isinstance(fk, dict)
        assert "strength" in fk

        s = fk["strength"]
        assert isinstance(s, GLMParams)
        assert isinstance(s.coef, float)
        assert s.coef == 0.7
        assert s.intercept is None

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test Ridge acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test Ridge acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1.0)
        with expectation:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "SVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test Ridge acceptable kwargs."""
        regularizer = self.cls()
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                    regularizer_strength=1.0,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=1.0,
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, should set to 1.0
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0
        # if changed to regularized, is set to None.
        model.regularizer = "UnRegularized"
        assert model.regularizer_strength is None

        # if changed back, should set to 1.0
        model.regularizer = "Ridge"

        assert model.regularizer_strength == 1.0

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_is_callable(self, loss):
        """Test Ridge callable loss."""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1.0)
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        runner(GLMParams(true_params.coef * 0.0, true_params.intercept), X, y)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        runner(
            GLMParams(
                jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
                true_params.intercept,
            ),
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_output_match(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64

        # set model params
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}

        model_bfgs = copy.deepcopy(model)
        model_bfgs.solver_name = "BFGS"

        init_pars = model.initialize_params(X, y)
        model.initialize_optimization_and_state(X, y, init_pars)

        init_pars_bfgs = model_bfgs.initialize_params(X, y)
        model_bfgs.initialize_optimization_and_state(X, y, init_pars_bfgs)

        runner_gd = model.optimization_run
        runner_bfgs = model_bfgs.optimization_run

        params_gd = runner_gd(
            GLMParams(true_params.coef * 0.0, true_params.intercept), X, y
        )[0]
        params_bfgs = runner_bfgs(
            GLMParams(true_params.coef * 0.0, true_params.intercept), X, y
        )[0]

        match_weights = np.allclose(params_gd.coef, params_bfgs.coef)
        match_intercepts = np.allclose(params_gd.intercept, params_bfgs.intercept)

        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Convex estimators should converge to the same numerical value."
            )

    @pytest.mark.requires_x64
    def test_solver_match_sklearn(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_kwargs = {"tol": 10**-12}
        model.solver_name = "BFGS"

        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner_bfgs = model.optimization_run
        params = runner_bfgs(
            GLMParams(true_params.coef * 0.0, true_params.intercept), X, y
        )[0]
        model_skl = PoissonRegressor(
            fit_intercept=True,
            tol=10**-12,
            alpha=model.regularizer_strength,
        )
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM solver estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["LBFGS", "ProximalGradient"])
    @pytest.mark.requires_x64
    def test_solver_match_sklearn_gamma(
        self, solver_name, gammaGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.inverse_link_function = jnp.exp
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_kwargs = {"tol": 10**-12}
        model.regularizer_strength = 0.1
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner_bfgs = model.optimization_run
        params = runner_bfgs(
            GLMParams(true_params.coef * 0.0, true_params.intercept), X, y
        )[0]
        model_skl = GammaRegressor(
            fit_intercept=True,
            tol=10**-12,
            alpha=0.1,
        )
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM estimate does not match sklearn!")

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "NonlinearCG",
            "ProximalGradient",
        ],
    )
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.fit(X, y)


class TestLasso:
    cls = nmo.regularizer.Lasso

    def test_filter_kwargs_contains_strength(self):
        """Test that strength is in filter kwargs."""
        n_features = 3
        params = GLMParams(coef=jnp.ones((n_features,)), intercept=jnp.array([0.0]))
        regularizer = self.cls()

        strength_vec = jnp.array([0.1, 0.2, 0.3])
        fk = regularizer._get_filter_kwargs(params=params, strength=strength_vec)
        assert isinstance(fk, dict)
        assert "strength" in fk

        s = fk["strength"]
        assert isinstance(s, GLMParams)
        assert isinstance(s.coef, jnp.ndarray)
        assert s.coef.shape == (3,)
        assert jnp.allclose(s.coef, strength_vec)
        assert s.intercept is None

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test Lasso acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test Lasso acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        with expectation:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_kwargs, solver_name):
        """Test LassoSolver acceptable kwargs."""
        regularizer = self.cls()
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                    regularizer_strength=1.0,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=1.0,
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, should set to 1.0
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        assert model.regularizer_strength is None

        # if changed back, should set to 1.0
        model.regularizer = regularizer

        assert model.regularizer_strength == 1.0

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        runner(GLMParams(true_params.coef * 0.0, true_params.intercept), X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        runner(
            GLMParams(
                jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
                true_params.intercept,
            ),
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_statsmodels(
        self, solver_name, poissonGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}

        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        params = runner(GLMParams(true_params.coef * 0.0, true_params.intercept), X, y)[
            0
        ]

        # instantiate the glm with statsmodels
        glm_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Poisson())

        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[1] + 1) * 1.0
        alpha_sm[0] = 0

        # pure lasso = elastic net with L1 weight = 1
        res_sm = glm_sm.fit_regularized(
            method="elastic_net", alpha=alpha_sm, L1_wt=1.0, cnvrg_tol=10**-12
        )
        # compare params
        sm_params = res_sm.params
        glm_params = jnp.hstack((params.intercept, params.coef.flatten()))
        match_weights = np.allclose(sm_params, glm_params)
        if not match_weights:
            raise ValueError("Lasso GLM solver estimate does not match statsmodels!")

    def test_lasso_pytree(self, poissonGLM_model_instantiation_pytree):
        """Check pytree X can be fit."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree
        model.set_params(regularizer=nmo.regularizer.Lasso(), regularizer_strength=1.0)
        model.solver_name = "ProximalGradient"
        model.fit(X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("reg_str", [0.001, 0.01, 0.1, 1, 10])
    @pytest.mark.requires_x64
    def test_lasso_pytree_match(
        self,
        reg_str,
        solver_name,
        poissonGLM_model_instantiation_pytree,
        poissonGLM_model_instantiation,
    ):
        """Check pytree and array find same solution."""
        X, _, model, _, _ = poissonGLM_model_instantiation_pytree
        X_array, y, model_array, _, _ = poissonGLM_model_instantiation

        model.set_params(
            regularizer=nmo.regularizer.Lasso(), regularizer_strength=reg_str
        )
        model_array.set_params(
            regularizer=nmo.regularizer.Lasso(), regularizer_strength=reg_str
        )
        model.solver_name = solver_name
        model_array.solver_name = solver_name
        model.fit(X, y)
        model_array.fit(X_array, y)
        assert np.allclose(
            np.hstack(jax.tree_util.tree_leaves(model.coef_)), model_array.coef_
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.fit(X, y)


class TestElasticNet:
    cls = nmo.regularizer.ElasticNet

    def test_filter_kwargs_contains_strength(self):
        """Test that strength is in filter kwargs."""
        n_features = 3
        params = GLMParams(coef=jnp.ones((n_features,)), intercept=jnp.array([0.0]))
        regularizer = self.cls()

        fk = regularizer._get_filter_kwargs(params=params, strength=(0.7, 0.4))
        assert isinstance(fk, dict)
        assert "strength" in fk

        s = fk["strength"]
        assert isinstance(s, GLMParams)
        assert isinstance(s.coef, tuple) and len(s.coef) == 2
        s_coef, r_coef = s.coef
        assert isinstance(s_coef, float) and s_coef == 0.7
        assert isinstance(r_coef, float) and r_coef == 0.4
        assert s.intercept is None

    @pytest.mark.parametrize(
        "strength, expectation, check_fn",
        [
            # None -> defaults to (1.0, 0.5)
            (
                None,
                does_not_raise(),
                lambda result: (
                    isinstance(result, tuple)
                    and isinstance(result[0], float)
                    and isinstance(result[1], float)
                    and result == (1.0, 0.5)
                ),
            ),
            # Single float -> (strength, 0.5)
            (
                0.6,
                does_not_raise(),
                lambda result: (
                    isinstance(result, tuple)
                    and isinstance(result[0], float)
                    and isinstance(result[1], float)
                    and result == (0.6, 0.5)
                ),
            ),
            (
                1.0,
                does_not_raise(),
                lambda result: (
                    isinstance(result, tuple)
                    and isinstance(result[0], float)
                    and isinstance(result[1], float)
                    and result == (1.0, 0.5)
                ),
            ),
            # 0-d arrays should become Python floats; ratio defaults to 0.5
            (
                jnp.array(1.0),
                does_not_raise(),
                lambda result: (
                    isinstance(result, tuple)
                    and isinstance(result[0], float)
                    and isinstance(result[1], float)
                    and result == (1.0, 0.5)
                ),
            ),
            (
                np.array(0.5),
                does_not_raise(),
                lambda result: (
                    isinstance(result, tuple)
                    and isinstance(result[0], float)
                    and isinstance(result[1], float)
                    and result == (0.5, 0.5)
                ),
            ),
            # Explicit (strength, ratio) tuple
            (
                (1.0, 1.0),
                does_not_raise(),
                lambda result: (
                    isinstance(result, tuple)
                    and isinstance(result[0], float)
                    and isinstance(result[1], float)
                    and result == (1.0, 1.0)
                ),
            ),
            # Invalid: strength not convertible
            (
                "bah",
                pytest.raises(
                    TypeError,
                    match="Could not convert regularizer strength to floats: bah",
                ),
                lambda _: True,
            ),
            # Invalid: ratio not convertible
            (
                (1.0, "bah"),
                pytest.raises(
                    TypeError,
                    match="Could not convert regularizer strength to floats: bah",
                ),
                lambda _: True,
            ),
            # Invalid: ratio outside (0, 1]
            (
                (1.0, 0.0),
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "ElasticNet regularization ratio must be in (0, 1], got 0.0"
                    ),
                ),
                lambda _: True,
            ),
        ],
    )
    def test_strength_single_input(self, strength, expectation, check_fn):
        """Test that ElasticNet accepts scalar/0-d inputs (or None) and returns (strength, ratio)."""
        regularizer = self.cls()
        with expectation:
            result = regularizer._validate_strength(strength)
            assert check_fn(result)

    @pytest.mark.parametrize(
        "strength, ratio, expectation, check_fn",
        [
            # Dicts with scalar leaves (Python floats)
            (
                {"a": 0.5, "b": 0.3},
                {"a": 0.7, "b": 1.0},
                does_not_raise(),
                lambda res: (
                    isinstance(res, tuple)
                    and isinstance(res[0], dict)
                    and isinstance(res[1], dict)
                    and isinstance(res[0]["a"], float)
                    and res[0]["a"] == 0.5
                    and isinstance(res[0]["b"], float)
                    and res[0]["b"] == 0.3
                    and isinstance(res[1]["a"], float)
                    and res[1]["a"] == 0.7
                    and isinstance(res[1]["b"], float)
                    and res[1]["b"] == 1.0
                ),
            ),
            # Dicts with 0-d arrays (should be converted to float)
            (
                {"a": jnp.array(0.5), "b": np.array(0.3)},
                {"a": jnp.array(1.0), "b": np.array(0.8)},
                does_not_raise(),
                lambda res: (
                    isinstance(res, tuple)
                    and isinstance(res[0]["a"], float)
                    and res[0]["a"] == 0.5
                    and isinstance(res[0]["b"], float)
                    and res[0]["b"] == 0.3
                    and isinstance(res[1]["a"], float)
                    and res[1]["a"] == 1.0
                    and isinstance(res[1]["b"], float)
                    and res[1]["b"] == 0.8
                ),
            ),
            # Dicts with arrays (should be preserved as jnp.ndarray)
            (
                {"a": jnp.array([0.1, 0.2, 0.3]), "b": np.array([0.5])},
                {"a": jnp.array([0.9, 0.8, 0.7]), "b": np.array([1.0])},
                does_not_raise(),
                lambda res: (
                    isinstance(res, tuple)
                    and isinstance(res[0]["a"], jnp.ndarray)
                    and res[0]["a"].shape == (3,)
                    and jnp.allclose(res[0]["a"], jnp.array([0.1, 0.2, 0.3]))
                    and isinstance(res[0]["b"], jnp.ndarray)
                    and res[0]["b"].shape == (1,)
                    and jnp.allclose(res[0]["b"], jnp.array([0.5]))
                    and isinstance(res[1]["a"], jnp.ndarray)
                    and res[1]["a"].shape == (3,)
                    and jnp.allclose(res[1]["a"], jnp.array([0.9, 0.8, 0.7]))
                    and isinstance(res[1]["b"], jnp.ndarray)
                    and res[1]["b"].shape == (1,)
                    and jnp.allclose(res[1]["b"], jnp.array([1.0]))
                ),
            ),
            # Nested dicts with mixed leaves
            (
                {
                    "a": {"x": 0.3, "y": 0.4},
                    "b": {"x": jnp.array([0.1, 0.2]), "y": 0.5},
                },
                {
                    "a": {"x": 0.9, "y": 0.8},
                    "b": {"x": jnp.array([0.7, 0.6]), "y": 1.0},
                },
                does_not_raise(),
                lambda res: (
                    isinstance(res, tuple)
                    and isinstance(res[0]["a"]["x"], float)
                    and res[0]["a"]["x"] == 0.3
                    and isinstance(res[0]["a"]["y"], float)
                    and res[0]["a"]["y"] == 0.4
                    and isinstance(res[0]["b"]["x"], jnp.ndarray)
                    and res[0]["b"]["x"].shape == (2,)
                    and isinstance(res[0]["b"]["y"], float)
                    and res[0]["b"]["y"] == 0.5
                    and isinstance(res[1]["a"]["x"], float)
                    and res[1]["a"]["x"] == 0.9
                    and isinstance(res[1]["a"]["y"], float)
                    and res[1]["a"]["y"] == 0.8
                    and isinstance(res[1]["b"]["x"], jnp.ndarray)
                    and res[1]["b"]["x"].shape == (2,)
                    and isinstance(res[1]["b"]["y"], float)
                    and res[1]["b"]["y"] == 1.0
                ),
            ),
            # Array strength with scalar ratio
            (
                jnp.array([0.1, 0.2, 0.3]),
                0.7,
                does_not_raise(),
                lambda res: (
                    isinstance(res, tuple)
                    and isinstance(res[0], jnp.ndarray)
                    and res[0].shape == (3,)
                    and jnp.allclose(res[0], jnp.array([0.1, 0.2, 0.3]))
                    and isinstance(res[1], float)
                    and res[1] == 0.7
                ),
            ),
            # 2-D array for both strength and ratio
            (
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([[0.9, 0.8], [0.7, 1.0]]),
                does_not_raise(),
                lambda res: (
                    isinstance(res, tuple)
                    and isinstance(res[0], jnp.ndarray)
                    and res[0].shape == (2, 2)
                    and jnp.allclose(res[0], jnp.array([[0.1, 0.2], [0.3, 0.4]]))
                    and isinstance(res[1], jnp.ndarray)
                    and res[1].shape == (2, 2)
                    and jnp.allclose(res[1], jnp.array([[0.9, 0.8], [0.7, 1.0]]))
                ),
            ),
            # Invalid: strength tree contains invalid leaf
            (
                {"a": "invalid", "b": 0.5},
                {"a": 0.9, "b": 1.0},
                pytest.raises(
                    TypeError, match="Could not convert regularizer strength to floats:"
                ),
                lambda _: True,
            ),
            # Invalid: ratio out of range (includes 0)
            (
                {"a": 0.5, "b": 0.3},
                {"a": 0.0, "b": 0.8},
                pytest.raises(
                    ValueError, match="ElasticNet regularization ratio must be in"
                ),
                lambda _: True,
            ),
        ],
    )
    def test_strength_tree_input(self, strength, ratio, expectation, check_fn):
        """Test that ElasticNet accepts tree strength/ratio with proper type conversion.

        Conversion rules:
        - 0-dim arrays or scalar numbers -> float
        - Arrays with shape (n,) or higher -> jnp.ndarray with same shape
        - Strings or other invalid types -> TypeError
        - Ratio must be in (0, 1] (broadcasted per leaf)
        """
        regularizer = self.cls()
        with expectation:
            result = regularizer._validate_strength((strength, ratio))
            assert check_fn(result)

    def test_validate_strength_structure_scalar_broadcast(self):
        """Test that ElasticNet broadcasts (strength, ratio) to parameter structure."""
        regularizer = self.cls()
        params = GLMParams(coef=jnp.ones((4,)), intercept=jnp.array([0.0]))

        structured = regularizer._validate_strength_structure(params, (1.0, 0.5))

        # coef gets a tuple (s, r), intercept remains None
        assert isinstance(structured, GLMParams)
        assert isinstance(structured.coef, tuple) and len(structured.coef) == 2
        s, r = structured.coef
        assert isinstance(s, float) and s == 1.0
        assert isinstance(r, float) and r == 0.5
        assert structured.intercept is None

    @pytest.mark.parametrize(
        "params, strength, ratio, expectation",
        [
            # GLMParams: no groups
            (
                GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
                0.1,  # scalar strength
                0.5,  # scalar ratio
                does_not_raise(),
            ),
            (
                GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
                jnp.array([0.1, 0.2, 0.3]),  # array strength matches shape
                0.5,  # scalar ratio
                does_not_raise(),
            ),
            (
                GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
                0.1,  # scalar strength
                jnp.array([0.9, 0.8, 0.7]),  # array ratio matches shape
                does_not_raise(),
            ),
            (
                GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
                jnp.array([0.1, 0.2]),  # strength not enough
                0.5,
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            (
                GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
                0.1,
                jnp.array([0.9, 0.8]),  # ratio not enough
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            (
                GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
                jnp.array([0.1, 0.2, 0.3, 0.4]),  # strength too many
                0.5,
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            (
                GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0])),
                0.1,
                jnp.array([0.9, 0.8, 0.7, 0.6]),  # ratio too many
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            # GLMParams: 2 groups in coef (dict)
            (
                GLMParams(
                    coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                    intercept=jnp.array([0.0]),
                ),
                {"a": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.1, 0.2])},  # match
                {"a": jnp.array([0.9, 0.8, 0.7]), "b": jnp.array([1.0, 0.6])},  # match
                does_not_raise(),
            ),
            (
                GLMParams(
                    coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                    intercept=jnp.array([0.0]),
                ),
                {"a": 0.1, "b": 0.1},  # scalar per group
                {"a": 0.7, "b": 1.0},  # scalar per group
                does_not_raise(),
            ),
            (
                GLMParams(
                    coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                    intercept=jnp.array([0.0]),
                ),
                {"a": 0.1, "b": jnp.array([0.1, 0.2])},  # mixed strength
                {"a": 0.9, "b": jnp.array([0.8, 1.0])},  # mixed ratio
                does_not_raise(),
            ),
            (
                GLMParams(
                    coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                    intercept=jnp.array([0.0]),
                ),
                {
                    "a": jnp.array([0.1, 0.2, 0.3]),
                    "b": jnp.array([0.1]),
                },  # strength not enough in 'b'
                {"a": jnp.array([0.9, 0.8, 0.7]), "b": 1.0},
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            (
                GLMParams(
                    coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                    intercept=jnp.array([0.0]),
                ),
                {"a": 0.1, "b": 0.1},
                {
                    "a": jnp.array([0.9, 0.8, 0.7]),
                    "b": jnp.array([1.0]),
                },  # ratio not enough in 'b'
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            (
                GLMParams(
                    coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                    intercept=jnp.array([0.0]),
                ),
                {
                    "a": jnp.array([0.1, 0.2, 0.3, 0.4]),
                    "b": 0.1,
                },  # strength too many in 'a'
                {"a": 0.9, "b": jnp.array([1.0, 0.6])},
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            (
                GLMParams(
                    coef={"a": jnp.ones((3,)), "b": jnp.ones((2,))},
                    intercept=jnp.array([0.0]),
                ),
                {"a": 0.1, "b": 0.1},
                {
                    "a": jnp.array([0.9, 0.8, 0.7, 0.6]),
                    "b": 1.0,
                },  # ratio too many in 'a'
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            # Arbitrary object with regularizable subtrees: no groups (TwoParams)
            (
                TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
                [
                    jnp.array([0.1, 0.2, 0.3]),
                    jnp.array([0.1, 0.2]),
                ],  # per-subtree strength arrays match
                0.7,  # scalar ratio
                does_not_raise(),
            ),
            (
                TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
                0.2,  # scalar strength
                [
                    jnp.array([0.9, 0.8, 0.7]),
                    jnp.array([1.0, 0.6]),
                ],  # per-subtree ratio arrays match
                does_not_raise(),
            ),
            (
                TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
                [
                    jnp.array([0.1, 0.2]),
                    jnp.array([0.1, 0.2]),
                ],  # strength not enough for 'one'
                0.6,
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            (
                TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
                0.2,
                [
                    jnp.array([0.9, 0.8]),
                    jnp.array([1.0, 0.6]),
                ],  # ratio not enough for 'one'
                pytest.raises(
                    ValueError,
                    match=r"Strength shape .* does not match parameter shape .*",
                ),
            ),
            # Count mismatch in per-subtree specification (either in strength or ratio)
            (
                TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
                [0.1],  # only one strength provided
                0.5,
                pytest.raises(ValueError, match=r"Expected .* strength values, got "),
            ),
            (
                TwoParams(one=jnp.ones((3,)), two=jnp.ones((2,))),
                0.1,
                [0.5],  # only one ratio provided
                pytest.raises(ValueError, match=r"Expected .* strength values, got "),
            ),
        ],
    )
    def test_validate_strength_structure_shape_mismatch(
        self, params, strength, ratio, expectation
    ):
        """ElasticNet: verify (strength, ratio) structure/shape against parameter structure.

        - Scalars broadcast to regularizable leaves; non-regularizable leaves remain None.
        - Non-scalar array leaves must match the corresponding parameter leaf shape.
        - For objects exposing multiple regularizable subtrees, supply per-subtree strengths (lists/tuples);
          a count mismatch raises ValueError.
        """
        reg = self.cls()
        with expectation:
            reg._validate_strength_structure(params, (strength, ratio))

    def test_validate_strength_structure_transform(self):
        """Test that ElasticNet  correctly transforms to parameter structure with (strength, ratio) leafs."""
        regularizer = self.cls()
        params = {
            "one": jnp.ones((3,)),
            "two": jnp.ones((2,)),
        }

        strength_tree = {
            "one": jnp.array([0.1, 0.2, 0.3]),
            "two": jnp.array([1.0, 2.0]),
        }
        ratio_tree = {
            "one": jnp.array([0.9, 0.8, 0.7]),
            "two": jnp.array([1.0, 0.6]),
        }
        structured = regularizer._validate_strength_structure(
            params, (strength_tree, ratio_tree)
        )

        assert isinstance(structured, dict)
        assert isinstance(structured["one"], tuple)
        s1, r1 = structured["one"]
        s2, r2 = structured["two"]
        assert jnp.allclose(s1, jnp.array([0.1, 0.2, 0.3]))
        assert jnp.allclose(r1, jnp.array([0.9, 0.8, 0.7]))
        assert jnp.allclose(s2, jnp.array([1.0, 2.0]))
        assert jnp.allclose(r2, jnp.array([1.0, 0.6]))

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test ElasticNet acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test ElasticNet acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=(1, 0.5))
        with expectation:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_kwargs, solver_name):
        """Test ElasticNetSolver acceptable kwargs."""
        regularizer = self.cls()
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                    regularizer_strength=(1.0, 0.5),
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=(1.0, 0.5),
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, set to (1.0, 0.5)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == (1.0, 0.5)

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        assert model.regularizer_strength is None

        # if changed back, set to (1.0, 0.5)
        model.regularizer = regularizer

        assert model.regularizer_strength == (1.0, 0.5)

    def test_regularizer_strength_float(self):
        """Assert regularizer ratio handled appropriately when only strength provided."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=0.6)
        assert model.regularizer_strength == (0.6, 0.5)

    @pytest.mark.parametrize(
        "regularizer_strength, expectation",
        [
            ((1.0, 0.5), does_not_raise()),
            ((1.0, 1.0), does_not_raise()),
            (
                (1.0, 0.0),
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"ElasticNet regularization ratio must be in (0, 1], got 0.0"
                    ),
                ),
            ),
            (
                (1.0, 1.1),
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"ElasticNet regularization ratio must be in (0, 1], got 1.1"
                    ),
                ),
            ),
            (
                (1.0, -0.1),
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"ElasticNet regularization ratio must be in (0, 1], got -0.1"
                    ),
                ),
            ),
            (
                (1.0, "bah"),
                pytest.raises(
                    TypeError,
                    match="Could not convert regularizer strength to floats: bah",
                ),
            ),
            (
                (1.0, 0.5, 0.1),
                pytest.raises(
                    TypeError,
                    match=re.escape(
                        "ElasticNet regularizer strength must be a tuple (strength, ratio)"
                    ),
                ),
            ),
        ],
    )
    def test_regularizer_ratio_setter(self, regularizer_strength, expectation):
        """Test that the regularizer ratio setter works as expected."""
        regularizer = self.cls()
        with expectation:
            nmo.glm.GLM(
                regularizer=regularizer, regularizer_strength=regularizer_strength
            )

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=(1, 0.5))
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        model.set_params(regularizer=self.cls(), regularizer_strength=(1, 0.5))
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        runner(GLMParams(true_params.coef * 0.0, true_params.intercept), X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=(1, 0.5))
        model.solver_name = solver_name
        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        runner(
            GLMParams(
                jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
                true_params.intercept,
            ),
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient"])
    @pytest.mark.parametrize("regularizer_strength", [1.0, 0.5, 0.1])
    @pytest.mark.parametrize("reg_ratio", [1.0, 0.5, 0.2])
    @pytest.mark.requires_x64
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
    def test_solver_match_statsmodels(
        self,
        solver_name,
        regularizer_strength,
        reg_ratio,
        poissonGLM_model_instantiation,
    ):
        """Test that different solvers converge to the same solution."""
        # with jax.disable_jit():
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.set_params(
            regularizer=self.cls(),
            regularizer_strength=(regularizer_strength, reg_ratio),
        )
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12, "maxiter": 10000}

        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        runner = model.optimization_run
        params = runner(GLMParams(true_params.coef * 0.0, true_params.intercept), X, y)[
            0
        ]

        model.fit(X, y)
        # instantiate the glm with statsmodels
        glm_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Poisson())

        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[1] + 1) * regularizer_strength
        alpha_sm[0] = 0

        # pure lasso = elastic net with L1 weight = 1
        res_sm = glm_sm.fit_regularized(
            method="elastic_net",
            alpha=alpha_sm,
            L1_wt=reg_ratio,
            cnvrg_tol=10**-12,
            zero_tol=1e-1000,
            maxiter=10000,
        )
        # compare params
        sm_params = res_sm.params
        glm_params = jnp.hstack((params.intercept, params.coef.flatten()))
        assert np.allclose(sm_params, glm_params)

    @pytest.mark.requires_x64
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
    def test_loss_convergence(self):
        """Test that penalized loss converges to the same value as statsmodels and the proximal operator."""
        # generate toy data
        np.random.seed(123)
        num_samples, num_features = 1000, 5
        X = np.random.normal(size=(num_samples, num_features))  # design matrix
        w = list(np.random.normal(size=(num_features,)))  # define some weights
        y = np.random.poisson(np.exp(X.dot(w)))  # observed counts

        # instantiate and fit GLM with ProximalGradient
        model_PG = nmo.glm.GLM(
            regularizer="ElasticNet",
            regularizer_strength=(1.0, 0.5),
            solver_name="ProximalGradient",
            solver_kwargs=dict(tol=10**-12, maxiter=10000),
        )
        model_PG.fit(X, y)
        glm_res = np.hstack((model_PG.intercept_, model_PG.coef_))

        # use the penalized loss function to solve optimization via Nelder-Mead
        penalized_loss = lambda p, x, y: model_PG.regularizer.penalized_loss(
            model_PG._compute_loss,
            params=GLMParams(
                p[1:],
                p[0].reshape(
                    1,
                ),
            ),
            strength=model_PG.regularizer_strength,
        )(
            GLMParams(
                p[1:],
                p[0].reshape(
                    1,
                ),
            ),
            x,
            y,
        )
        res = minimize(
            penalized_loss,
            [0] + w,
            args=(X, y),
            method="Nelder-Mead",
            tol=10**-12,
            options={"maxiter": 10000},
        )
        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[1] + 1) * 1.0
        alpha_sm[0] = 0

        # elastic net with
        glm_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Poisson())
        res_sm = glm_sm.fit_regularized(
            method="elastic_net",
            alpha=alpha_sm,
            L1_wt=0.5,
            cnvrg_tol=10**-12,
            zero_tol=1e-1000,
            maxiter=10000,
        )
        # assert weights are the same
        assert np.allclose(res.x, glm_res)
        assert np.allclose(res.x, res_sm.params)
        assert np.allclose(glm_res, res_sm.params)

    def test_elasticnet_pytree(self, poissonGLM_model_instantiation_pytree):
        """Check pytree X can be fit."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree
        model.set_params(
            regularizer=nmo.regularizer.ElasticNet(), regularizer_strength=1.0
        )
        model.solver_name = "ProximalGradient"
        model.fit(X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("reg_str", [0.001, 0.01, 0.1, 1, 10])
    @pytest.mark.requires_x64
    def test_elasticnet_pytree_match(
        self,
        reg_str,
        solver_name,
        poissonGLM_model_instantiation_pytree,
        poissonGLM_model_instantiation,
    ):
        """Check pytree and array find same solution."""
        X, _, model, _, _ = poissonGLM_model_instantiation_pytree
        X_array, y, model_array, _, _ = poissonGLM_model_instantiation

        model.set_params(
            regularizer=nmo.regularizer.ElasticNet(), regularizer_strength=reg_str
        )
        model_array.set_params(
            regularizer=nmo.regularizer.ElasticNet(), regularizer_strength=reg_str
        )
        model.solver_name = solver_name
        model_array.solver_name = solver_name
        model.fit(X, y)
        model_array.fit(X_array, y)
        assert np.allclose(
            np.hstack(jax.tree_util.tree_leaves(model.coef_)), model_array.coef_
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.fit(X, y)


class TestGroupLasso:
    cls = nmo.regularizer.GroupLasso

    def test_filter_kwargs_contains_strength(self):
        """Test that strength is in filter kwargs."""
        n_features = 3
        # 2 groups x 3 features (coef), intercept leaf is None in the mask
        mask = GLMParams(
            coef=jnp.array([[1, 1, 0], [0, 0, 1]], dtype=float),
            intercept=None,
        )
        params = GLMParams(coef=jnp.ones((n_features,)), intercept=jnp.array([0.0]))
        regularizer = self.cls()

        fk = regularizer._get_filter_kwargs(params=params, strength=0.5)
        assert isinstance(fk, dict)
        assert "mask" in fk
        assert "strength" in fk

        s = fk["strength"]
        assert isinstance(s, GLMParams)
        assert isinstance(s.coef, jnp.ndarray)
        assert s.coef == 0.5

    def test_validate_strength_structure_scalar_broadcast(self):
        """Scalar strength broadcasts to per-group vector with length n_groups."""
        # 3 groups x 5 features mask on coef, intercept is not regularized (None)
        mask = GLMParams(
            coef=jnp.array(
                [
                    [1, 1, 1, 0, 0],  # group 0
                    [0, 0, 0, 1, 0],  # group 1
                    [0, 0, 0, 0, 1],  # group 2
                ],
                dtype=float,
            ),
            intercept=None,
        )
        params = GLMParams(coef=jnp.ones((5,)), intercept=jnp.array([0.0]))
        regularizer = self.cls(mask=mask)

        strength_struct = regularizer._validate_strength_structure(params, 0.7)
        # Returns a tree aligned to mask/params; coef must be per-group vector
        assert isinstance(strength_struct, GLMParams)
        assert isinstance(strength_struct.coef, jnp.ndarray)
        assert strength_struct.coef.shape == (3,)
        assert jnp.allclose(strength_struct.coef, jnp.array([0.7, 0.7, 0.7]))
        assert strength_struct.intercept is None

    def test_validate_strength_structure_shape_match(self):
        """Per-group vector matches n_groups length."""
        n_groups = 2
        mask = GLMParams(
            coef=jnp.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float),  # 2 groups
            intercept=None,
        )
        params = GLMParams(coef=jnp.ones((4,)), intercept=jnp.array([0.0]))
        regularizer = self.cls(mask=mask)

        per_group = jnp.array([0.1, 0.9])
        strength_struct = regularizer._validate_strength_structure(params, per_group)
        assert isinstance(strength_struct.coef, jnp.ndarray)
        assert strength_struct.coef.shape == (n_groups,)
        assert jnp.allclose(strength_struct.coef, per_group)
        assert strength_struct.intercept is None

    def test_validate_strength_structure_shape_mismatch(self):
        """Vector length must equal n_groups; otherwise ValueError."""
        n_groups = 3
        mask = GLMParams(
            coef=jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            intercept=None,
        )
        params = GLMParams(coef=jnp.ones((3,)), intercept=jnp.array([0.0]))
        regularizer = self.cls(mask=mask)

        too_short = jnp.array([0.5, 0.5])
        too_long = jnp.array([0.5, 0.5, 0.5, 0.5])

        with pytest.raises(
            ValueError,
            match=rf"GroupLasso strength must be a scalar or shape \({n_groups},\), got shape \({too_short.shape[0]},\)",
        ):
            regularizer._validate_strength_structure(params, too_short)

        with pytest.raises(
            ValueError,
            match=rf"GroupLasso strength must be a scalar or shape \({n_groups},\), got shape \({too_long.shape[0]},\)",
        ):
            regularizer._validate_strength_structure(params, too_long)

    def test_validate_strength_structure_dict(self):
        """Dict/PyTree mask: scalar strength broadcasts to per-group vector for each leaf."""
        # Mask as dict with consistent n_groups across leaves
        mask = {
            "a": jnp.array([[1, 1, 0], [0, 0, 1]], dtype=float),  # 2 groups, 3 features
            "b": jnp.array([[1, 0], [0, 1]], dtype=float),  # 2 groups, 2 features
        }
        regularizer = self.cls(mask=mask)
        params = {
            "a": jnp.ones((3,)),
            "b": jnp.ones((2,)),
        }

        strength_struct = regularizer._validate_strength_structure(params, 0.2)
        # Should mirror mask structure with per-group vectors
        assert isinstance(strength_struct, dict)
        assert isinstance(strength_struct["a"], jnp.ndarray)
        assert isinstance(strength_struct["b"], jnp.ndarray)
        assert strength_struct["a"].shape == (2,) and jnp.allclose(
            strength_struct["a"], jnp.array([0.2, 0.2])
        )
        assert strength_struct["b"].shape == (2,) and jnp.allclose(
            strength_struct["b"], jnp.array([0.2, 0.2])
        )

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test GroupLasso acceptable solvers."""
        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(mask=mask), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test GroupLassoSolver acceptable solvers."""
        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)
        regularizer = self.cls(mask=mask)
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        with expectation:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test GroupLasso acceptable kwargs."""
        raise_exception = "tols" in list(solver_kwargs.keys())

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[0, 1:] = 1
        mask = jnp.asarray(mask)

        regularizer = self.cls(mask=mask)

        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                    regularizer_strength=1.0,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=1.0,
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, should set to 1.0
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        assert model.regularizer_strength is None

        # if changed back, should set to 1.0
        model.regularizer = regularizer

        assert model.regularizer_strength == 1.0

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {"mask": None}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)

        regularizer = self.cls(mask=mask)
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1.0)
        model._compute_loss = loss

        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask with new PyTree structure
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = GLMParams(jnp.asarray(mask), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)
        model.optimization_run(
            GLMParams(true_params.coef * 0.0, true_params.intercept), X, y
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_init_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver initialization returns a state."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask with new PyTree structure
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = GLMParams(jnp.asarray(mask), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)
        state = model.optimization_init_state(true_params, X, y)
        # asses that state is a NamedTuple by checking tuple type and the availability of some NamedTuple
        # specific namespace attributes
        assert isinstance(state, tuple | eqx.Module)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_update_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver initialization returns a state."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask with new PyTree structure
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = GLMParams(jnp.asarray(mask), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        init_pars = model.initialize_params(X, y)

        model.initialize_optimization_and_state(X, y, init_pars)

        state = model.optimization_init_state(
            GLMParams(true_params.coef * 0.0, true_params.intercept), X, y
        )

        # ProxSVRG needs the full gradient at the anchor point to be initialized
        # so here just set it to xs, which is not correct, but fine shape-wise
        if solver_name == "ProxSVRG":
            state = state._replace(full_grad_at_reference_point=state.reference_point)

        params, state, _ = model.optimization_update(true_params, state, X, y)

        # asses that state is a NamedTuple by checking tuple type and the availability of some NamedTuple
        # specific namespace attributes
        assert isinstance(state, tuple | eqx.Module)

        # check params struct and shapes
        assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
            true_params
        )
        assert all(
            jax.tree_util.tree_leaves(params)[k].shape == p.shape
            for k, p in enumerate(jax.tree_util.tree_leaves(true_params))
        )

    @pytest.mark.parametrize("n_groups_assign", [0, 1, 2])
    def test_mask_validity_groups(
        self, n_groups_assign, poissonGLM_model_instantiation_group_sparse
    ):
        """Test that mask assigns at most 1 group to each weight."""
        raise_exception = n_groups_assign > 1
        (
            X,
            y,
            model,
            true_params,
            firing_rate,
            _,
        ) = poissonGLM_model_instantiation_group_sparse

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1

        # change assignment
        if n_groups_assign == 0:
            mask[:, 3] = 0
        elif n_groups_assign == 2:
            mask[:, 3] = 1

        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(
                ValueError, match="Incorrect group assignment. Some of the features"
            ):
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

    @pytest.mark.parametrize("set_entry", [0, 1, -1, 2, 2.5])
    def test_mask_validity_entries(self, set_entry, poissonGLM_model_instantiation):
        """Test that mask is composed of 0s and 1s."""
        raise_exception = set_entry not in {0, 1}
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        # assign an entry
        mask[1, 2] = set_entry
        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match="Mask elements must be 0s and 1s"):
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

    @pytest.mark.parametrize("n_dim", [0, 1, 2, 3])
    def test_mask_dimension_1(self, n_dim, poissonGLM_model_instantiation):
        """Test that mask works with PyTree structure."""

        # With PyTree masks, we need proper structure
        raise_exception = n_dim in [0, 1]
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create masks with different dimensions
        if n_dim == 0:
            mask = np.array([])
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 1:
            mask = np.ones((1,))
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 2:
            # Valid PyTree mask structure
            mask = np.zeros((2, X.shape[1]))
            mask[0, :2] = 1
            mask[1, 2:] = 1
            mask = GLMParams(jnp.asarray(mask, dtype=jnp.float32), None)
        else:
            # 3D mask needs to be wrapped in PyTree
            mask = np.zeros((2, X.shape[1]) + (1,) * (n_dim - 2))
            mask[0, :2] = 1
            mask[1, 2:] = 1
            mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError):
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

    @pytest.mark.parametrize("n_groups", [0, 1, 2])
    def test_mask_n_groups(self, n_groups, poissonGLM_model_instantiation):
        """Test that mask has at least 1 group."""
        raise_exception = n_groups < 1
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a mask with PyTree structure
        mask_array = np.zeros((n_groups, X.shape[1]))
        if n_groups > 0:
            for i in range(n_groups - 1):
                mask_array[i, i : i + 1] = 1
            mask_array[-1, n_groups - 1 :] = 1

        mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)

        if raise_exception:
            with pytest.raises(ValueError, match=r"Empty mask provided!"):
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

    def test_group_sparsity_enforcement(
        self, poissonGLM_model_instantiation_group_sparse
    ):
        """Test that group lasso works on a simple dataset."""
        (
            X,
            y,
            model,
            true_params,
            firing_rate,
            _,
        ) = poissonGLM_model_instantiation_group_sparse
        zeros_true = true_params.coef.flatten() == 0
        mask_array = np.zeros((2, X.shape[1]))
        mask_array[0, zeros_true] = 1
        mask_array[1, ~zeros_true] = 1
        mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = "ProximalGradient"

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner = model._instantiate_solver(model._compute_loss, init_params)[-1]
        params, _, _ = runner(
            init_params, X, y
        )

        zeros_est = params.coef == 0.0
        if not np.all(zeros_est == zeros_true):
            raise ValueError("GroupLasso failed to zero-out the parameter group!")

    ###########
    # Test mask from set_params
    ###########
    @pytest.mark.parametrize("n_groups_assign", [0, 1, 2])
    def test_mask_validity_groups_set_params(
        self, n_groups_assign, poissonGLM_model_instantiation_group_sparse
    ):
        """Test that mask assigns at most 1 group to each weight."""
        raise_exception = n_groups_assign > 1
        (
            X,
            y,
            model,
            true_params,
            firing_rate,
            _,
        ) = poissonGLM_model_instantiation_group_sparse

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        regularizer = self.cls(mask=mask)

        # change assignment
        if n_groups_assign == 0:
            mask[:, 3] = 0
        elif n_groups_assign == 2:
            mask[:, 3] = 1

        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(
                ValueError, match="Incorrect group assignment. Some of the features"
            ):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("set_entry", [0, 1, -1, 2, 2.5])
    def test_mask_validity_entries_set_params(
        self, set_entry, poissonGLM_model_instantiation
    ):
        """Test that mask is composed of 0s and 1s."""
        raise_exception = set_entry not in {0, 1}
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        regularizer = self.cls(mask=mask)

        # assign an entry
        mask[1, 2] = set_entry
        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match="Mask elements must be 0s and 1s"):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("n_dim", [0, 1, 2, 3])
    def test_mask_dimension(self, n_dim, poissonGLM_model_instantiation):
        """Test that mask works with PyTree structure."""

        raise_exception = n_dim in [0, 1]
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        valid_mask_array = np.zeros((2, X.shape[1]))
        valid_mask_array[0, :1] = 1
        valid_mask_array[1, 1:] = 1
        valid_mask = GLMParams(jnp.asarray(valid_mask_array, dtype=jnp.float32), None)
        regularizer = self.cls(mask=valid_mask)

        # create masks with different dimensions
        if n_dim == 0:
            mask = np.array([])
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 1:
            mask = np.ones((1,))
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 2:
            mask_array = np.zeros((2, X.shape[1]))
            mask_array[0, :2] = 1
            mask_array[1, 2:] = 1
            mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)
        else:
            mask = np.zeros((2, X.shape[1]) + (1,) * (n_dim - 2))
            mask[0, :2] = 1
            mask[1, 2:] = 1
            mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("n_groups", [0, 1, 2])
    def test_mask_n_groups_set_params(self, n_groups, poissonGLM_model_instantiation):
        """Test that mask has at least 1 group."""
        raise_exception = n_groups < 1
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        valid_mask_array = np.zeros((2, X.shape[1]))
        valid_mask_array[0, :1] = 1
        valid_mask_array[1, 1:] = 1
        valid_mask = GLMParams(jnp.asarray(valid_mask_array, dtype=jnp.float32), None)
        regularizer = self.cls(mask=valid_mask)

        # create a mask with PyTree structure
        mask_array = np.zeros((n_groups, X.shape[1]))
        if n_groups > 0:
            for i in range(n_groups - 1):
                mask_array[i, i : i + 1] = 1
            mask_array[-1, n_groups - 1 :] = 1

        mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)

        if raise_exception:
            with pytest.raises(ValueError, match=r"Empty mask provided!"):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    def test_mask_none(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # Test with auto-initialized mask (mask=None, initialized during fit)
        model.regularizer = self.cls(mask=None)
        model.solver_name = "ProximalGradient"
        model.fit(X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        mask_array = np.ones((1, X.shape[1])).astype(float)
        mask = GLMParams(jnp.asarray(mask_array), None)
        model.set_params(
            regularizer=self.cls(mask=mask),
            regularizer_strength=(
                None if self.cls == nmo.regularizer.UnRegularized else 1.0
            ),
        )
        model.solver_name = solver_name
        model.fit(X, y)

    @pytest.mark.parametrize(
        "params_factory,expected_type,check_mask_fn",
        [
            # GLMParams single neuron (with regularizable_subtrees)
            (
                lambda: GLMParams(coef=jnp.ones((10, 3)), intercept=jnp.zeros(3)),
                GLMParams,
                lambda mask: (
                    mask.coef is not None
                    and mask.intercept is None
                    and mask.coef.ndim == 3
                    and mask.coef.shape[1:] == (10, 3)
                ),
            ),
            # Plain dict (without regularizable_subtrees)
            (
                lambda: {"spatial": jnp.ones((5, 2)), "temporal": jnp.ones((3, 2))},
                dict,
                lambda mask: (
                    "spatial" in mask
                    and "temporal" in mask
                    and mask["spatial"].ndim == 3
                    and mask["spatial"].shape[1:] == (5, 2)
                    and mask["temporal"].ndim == 3
                    and mask["temporal"].shape[1:] == (3, 2)
                ),
            ),
            # GLMParams multi-neuron (PopulationGLM case)
            (
                lambda: GLMParams(coef=jnp.ones((10, 5)), intercept=jnp.zeros(5)),
                GLMParams,
                lambda mask: (
                    mask.coef is not None
                    and mask.intercept is None
                    and mask.coef.ndim == 3
                    and mask.coef.shape[1:] == (10, 5)
                    and mask.coef.shape[0] == 5  # 5 groups (one per neuron)
                ),
            ),
        ],
    )
    def test_initialize_mask_different_structures(
        self, params_factory, expected_type, check_mask_fn
    ):
        """Test mask initialization for different parameter structures."""
        params = params_factory()
        regularizer = self.cls(mask=None)
        mask = regularizer.initialize_mask(params)

        # Check mask has expected type
        assert isinstance(mask, expected_type)

        # Check structure-specific properties
        assert check_mask_fn(mask)

    def test_apply_operator_dict_structure(self):
        """Test apply_operator with dict-based PyTree parameters."""
        from nemos.regularizer import apply_operator

        # Define a simple operation that doubles values
        def double_func(x):
            return jax.tree_util.tree_map(lambda a: a * 2, x)

        # Test with dict structure (no regularizable_subtrees)
        params = {
            "coef": jnp.ones((5,)),
            "bias": jnp.zeros((1,)),
        }

        result = apply_operator(double_func, params)

        # Check structure preserved
        assert isinstance(result, dict)
        assert set(result.keys()) == {"coef", "bias"}

        # Check operation was applied
        assert jnp.allclose(result["coef"], jnp.ones((5,)) * 2)
        assert jnp.allclose(result["bias"], jnp.zeros((1,)) * 2)

    def test_apply_operator_with_filter_kwargs(self):
        """Test apply_operator with filter_kwargs for routing masks."""
        from nemos.regularizer import apply_operator

        # Create GLMParams with regularizable_subtrees
        params = GLMParams(
            coef=jnp.ones((5,)),
            intercept=jnp.zeros((1,)),
        )

        # Create mask with matching structure
        mask = GLMParams(
            coef=jnp.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=float),
            intercept=None,
        )

        # Define a function that uses the mask to check it's passed correctly
        def masked_operation(x, mask=None):
            # Return a marker value to verify mask is passed/not passed
            if mask is None:
                return x * 0  # Return zeros if no mask
            else:
                return x * 2  # Return doubled if mask is present

        result = apply_operator(masked_operation, params, filter_kwargs={"mask": mask})

        # Check that mask was correctly routed to coef but not intercept
        # coef should be doubled (mask was passed)
        assert jnp.allclose(result.coef, params.coef * 2)
        # intercept should be zeros (no mask, returned x * 0)
        assert jnp.allclose(result.intercept, jnp.zeros((1,)))

    def test_penalized_loss_dict_structure(self, poissonGLM_model_instantiation):
        """Test penalized_loss with dict-based PyTree parameters."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # Create dict-based mask (simulating FeaturePytree structure)
        # Split features into two groups
        n_features = X.shape[1]
        mask_dict = {
            "group1": jnp.array([[1] * (n_features // 2)], dtype=float),
            "group2": jnp.array([[1] * (n_features - n_features // 2)], dtype=float),
        }

        # Note: For this test we're just checking that the method can be called
        # with dict structure, not testing actual GLM fitting
        regularizer = self.cls(mask=mask_dict)

        # Create matching dict params
        params_dict = {
            "group1": jnp.ones((n_features // 2,)),
            "group2": jnp.ones((n_features - n_features // 2,)),
        }

        # Test that penalization doesn't crash with dict structure
        filter_kwargs = regularizer._get_filter_kwargs(
            params_dict, strength=regularizer._validate_strength(0.1)
        )
        penalty = regularizer._penalization(params_dict, filter_kwargs=filter_kwargs)

        # Check penalty is a scalar and non-negative
        assert isinstance(penalty, jnp.ndarray)
        assert penalty.ndim == 0
        assert penalty >= 0

    def test_penalized_loss_glmparams_structure(self, poissonGLM_model_instantiation):
        """Test penalized_loss with GLMParams structure."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # Create GLMParams mask
        n_features = X.shape[1]
        mask_array = np.ones((2, n_features), dtype=float)
        mask_array[0, n_features // 2 :] = 0
        mask_array[1, : n_features // 2] = 0
        mask = GLMParams(jnp.asarray(mask_array), None)

        regularizer = self.cls(mask=mask)

        # Create matching GLMParams params
        params = GLMParams(
            coef=jnp.ones((n_features,)),
            intercept=jnp.zeros((1,)),
        )

        # Test that penalization works
        filter_kwargs = regularizer._get_filter_kwargs(
            params, strength=regularizer._validate_strength(0.1)
        )
        penalty = regularizer._penalization(params, filter_kwargs=filter_kwargs)

        # Check penalty is a scalar and non-negative
        assert isinstance(penalty, jnp.ndarray)
        assert penalty.ndim == 0
        assert penalty >= 0


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.UnRegularized(),
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso(mask=GLMParams(jnp.eye(5, dtype=jnp.float32), None)),
        nmo.regularizer.ElasticNet(),
    ],
)
class TestPenalizedLossAuxiliaryVariables:
    """Test that penalized_loss correctly handles auxiliary variables."""

    def test_single_value_return(self, regularizer):
        """Test backward compatibility: loss returning single value."""

        def simple_loss(params, X, y):
            return jnp.mean((y - X @ params.coef - params.intercept) ** 2)

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        # ElasticNet requires (strength, ratio) tuple
        regularizer_strength = regularizer._validate_strength(
            (0.1, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 0.1
        )
        penalized = regularizer.penalized_loss(
            simple_loss, params=params, strength=regularizer_strength
        )

        result = penalized(params, X, y)

        # Should return a single scalar value
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_tuple_return_with_aux(self, regularizer):
        """Test that loss returning (loss, aux) preserves auxiliary variable."""

        def loss_with_aux(params, X, y):
            predictions = X @ params.coef + params.intercept
            loss = jnp.mean((y - predictions) ** 2)
            aux = {"predictions": predictions, "mse": loss}
            return loss, aux

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        # ElasticNet requires (strength, ratio) tuple
        regularizer_strength = regularizer._validate_strength(
            (1.0, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 1.0
        )
        penalized = regularizer.penalized_loss(
            loss_with_aux, params=params, strength=regularizer_strength
        )

        result = penalized(params, X, y)

        # Should return a tuple (penalized_loss, aux)
        assert isinstance(result, tuple)
        assert len(result) == 2

        penalized_loss_value, aux = result

        # Check that penalized loss is a scalar
        assert isinstance(penalized_loss_value, jnp.ndarray)
        assert penalized_loss_value.shape == ()
        assert jnp.isfinite(penalized_loss_value)

        # Check that auxiliary variable is preserved
        assert isinstance(aux, dict)
        assert "predictions" in aux
        assert "mse" in aux
        assert aux["predictions"].shape == (10,)

        # Check that penalized loss > original loss (penalty added)
        if not isinstance(regularizer, nmo.regularizer.UnRegularized):
            assert penalized_loss_value > aux["mse"]

    def test_invalid_tuple_single_element(self, regularizer):
        """Test that single-element tuple raises error."""

        def bad_loss(params, X, y):
            return (jnp.mean((y - X @ params.coef - params.intercept) ** 2),)

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        # ElasticNet requires (strength, ratio) tuple
        regularizer_strength = regularizer._validate_strength(
            (1.0, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 1.0
        )
        penalized = regularizer.penalized_loss(
            bad_loss, params=params, strength=regularizer_strength
        )

        with pytest.raises(
            ValueError,
            match=r"Invalid loss function return.*returns a tuple with 1 value",
        ):
            penalized(params, X, y)

    def test_invalid_tuple_three_elements(self, regularizer):
        """Test that 3+ element tuple raises error."""

        def bad_loss(params, X, y):
            loss = jnp.mean((y - X @ params.coef - params.intercept) ** 2)
            return loss, {"aux": 1}, {"extra": 2}

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        # ElasticNet requires (strength, ratio) tuple
        regularizer_strength = regularizer._validate_strength(
            (1.0, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 1.0
        )
        penalized = regularizer.penalized_loss(
            bad_loss, params=params, strength=regularizer_strength
        )

        with pytest.raises(
            ValueError,
            match=r"Invalid loss function return.*returns a tuple with 3 values",
        ):
            penalized(params, X, y)

    def test_penalty_correctly_added_to_loss_with_aux(self, regularizer):
        """Test that penalty is correctly added when aux variables are present."""

        def loss_with_aux(params, X, y):
            predictions = X @ params.coef + params.intercept
            loss = jnp.mean((y - predictions) ** 2)
            return loss, {"predictions": predictions}

        # Get unpenalized loss
        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.zeros(10)

        unpenalized_loss, _ = loss_with_aux(params, X, y)

        # ElasticNet requires (strength, ratio) tuple
        regularizer_strength = regularizer._validate_strength(
            (1.0, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 1.0
        )

        # Get penalized loss
        penalized = regularizer.penalized_loss(
            loss_with_aux,
            params=params,
            strength=regularizer_strength,
        )
        penalized_loss_value, aux = penalized(params, X, y)

        # Calculate expected penalty
        filter_kwargs = regularizer._get_filter_kwargs(
            strength=regularizer_strength, params=params
        )
        expected_penalty = regularizer._penalization(params, filter_kwargs)

        # Check that penalized loss = unpenalized loss + penalty
        assert jnp.isclose(penalized_loss_value, unpenalized_loss + expected_penalty)


def test_available_regularizer_match():
    """Test matching of the two regularizer lists."""
    assert set(nmo._regularizer_builder.AVAILABLE_REGULARIZERS) == set(
        nmo.regularizer.__dir__()
    )
