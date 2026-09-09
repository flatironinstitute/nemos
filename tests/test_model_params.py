"""Meta-tests for ``ModelParams`` containers.

``Regularizer.resolve_hess_tag`` decides regularizer coverage by comparing leaf
ids between ``where(params)`` (for each ``where`` in ``regularizable_subtrees``)
and the full parameter tree. A ``where`` that returns transformed copies instead
of the stored leaves silently downgrades the Hessian tag to ``General``, forcing
the Newton solver onto the SVD path instead of Cholesky. The tests below walk
every ``ModelParams`` subclass and assert the identity-preservation invariant,
so any future params class that breaks it fails loudly here.
"""

import dataclasses
import importlib
import itertools
import pkgutil
import typing

import jax
import jax.numpy as jnp
import pytest

import nemos
from conftest import all_subclasses
from nemos._inspect_utils import is_abstract
from nemos.glm.params import GLMParams
from nemos.params import ModelParams
from nemos.regularizer import Ridge, UnRegularized
from nemos.solvers._hess import Diagonal, General, HessianTag, PositiveDefinite

# Import every submodule so all ModelParams subclasses are registered before the
# parametrizations below are collected (same idiom as test_hmm_validator).
for _, _modname, _ in pkgutil.walk_packages(nemos.__path__, prefix="nemos."):
    importlib.import_module(_modname)

_FILL = itertools.count()


def _field_hints(cls) -> dict:
    """Resolved field annotations for a params class.

    Falls back to the raw (possibly string) annotations when a
    TYPE_CHECKING-only name makes resolution fail; unresolved fields are then
    treated as array leaves by ``_dummy_instance``.
    """
    try:
        return typing.get_type_hints(cls)
    except NameError:
        return {f.name: f.type for f in dataclasses.fields(cls)}


def _dummy_instance(cls: type) -> ModelParams:
    """Build a params instance with one distinct array per leaf field.

    Fields annotated with a ``ModelParams`` subclass are built recursively;
    every other field gets a fresh array with unique contents so leaf ids are
    distinct across the whole tree.
    """
    hints = _field_hints(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        annotation = hints.get(field.name, jnp.ndarray)
        candidates = typing.get_args(annotation) or (annotation,)
        nested = next(
            (
                cand
                for cand in candidates
                if isinstance(cand, type) and issubclass(cand, ModelParams)
            ),
            None,
        )
        if nested is not None:
            kwargs[field.name] = _dummy_instance(nested)
        else:
            offset = next(_FILL)
            kwargs[field.name] = jnp.arange(offset, offset + 3, dtype=float)
    return cls(**kwargs)


_ALL_PARAMS_CLASSES = sorted(
    (cls for cls in all_subclasses(ModelParams) if not is_abstract(cls)),
    key=lambda cls: cls.__name__,
)


@pytest.mark.parametrize("params_cls", _ALL_PARAMS_CLASSES)
def test_regularizable_subtrees_preserve_leaf_identity(params_cls):
    """``where(params)`` must return the stored leaves, not transformed copies."""
    params = _dummy_instance(params_cls)
    all_ids = {id(leaf) for leaf in jax.tree_util.tree_leaves(params)}
    # make sure ids are unique
    assert len(all_ids) == len(jax.tree_util.tree_leaves(params))
    for where in params_cls.regularizable_subtrees():
        leaves = jax.tree_util.tree_leaves(where(params))
        missing = [type(leaf).__name__ for leaf in leaves if id(leaf) not in all_ids]
        assert not missing, (
            f"{params_cls.__name__}.regularizable_subtrees returned leaves that are "
            f"not identical (by id) to the stored params ({missing}); "
            "Regularizer.resolve_hess_tag coverage detection relies on identity "
            "preservation."
        )


class _FullyCoveredParams(ModelParams):
    """Params whose single subtree covers every leaf."""

    weights: jnp.ndarray

    @staticmethod
    def regularizable_subtrees():
        return [lambda p: p.weights]


@pytest.mark.parametrize(
    "params_cls, expected_property",
    [
        pytest.param(
            _FullyCoveredParams, PositiveDefinite, id="full-coverage-keeps-tag"
        ),
        pytest.param(GLMParams, General, id="partial-coverage-downgrades"),
    ],
)
def test_resolve_hess_tag_coverage(params_cls, expected_property):
    """Ridge's tag survives full coverage and downgrades on partial coverage."""
    params = _dummy_instance(params_cls)
    tag = Ridge().resolve_hess_tag(params)
    assert tag == HessianTag(structure=Diagonal, property=expected_property)


def test_resolve_hess_tag_unregularized_is_none():
    """A regularizer without a Hessian tag resolves to None."""
    params = _dummy_instance(GLMParams)
    assert UnRegularized().resolve_hess_tag(params) is None
