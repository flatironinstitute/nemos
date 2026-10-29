from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from jaxtyping import Array, PyTree

from ..._hess import HessianTag
from ._typing import R, S, Y

if TYPE_CHECKING:
    from ._typing import HvpFn
from ._utils import map_blocks


class AbstractCurvature(eqx.Module, ABC, Generic[Y, S, R]):
    """A model of the loss curvature, in whichever form the model has it.

    ``update`` hands back three things: a Hessian-vector product, usable by any
    direction; the assembled Hessian, or ``None`` when the model never assembles one;
    and the state to carry into the next iteration.

    ``S`` and ``R`` are separate because they coincide for only one of the two models:
    L-BFGS represents B_k by the same pair history it carries forward, while Newton
    carries nothing and represents it by the Hessian it has just assembled.
    """

    @abstractmethod
    def init(self, params: Y, *args: Any) -> S: ...

    @abstractmethod
    def update(
        self,
        state: S,
        params: Y,
        y_diff: Y,
        grad: Y,
        *args: Any,
    ) -> tuple[HvpFn, PyTree[Array] | None, S]: ...

    @abstractmethod
    def hvp(
        self, hessian_repr: R, params: Y, v: Y, hessian_tag: HessianTag | None
    ) -> Y: ...


def _batched_tree_zeros_like(y: Y, batch_dimension: int) -> PyTree[Array]:
    return jtu.tree_map(lambda y: jnp.zeros((batch_dimension, *y.shape)), y)


v_tree_dot = jax.vmap(lx.internal.tree_dot, in_axes=(0, None), out_axes=0)


class NewtonCurvature(AbstractCurvature[Y, None, PyTree[Array]], Generic[Y]):
    """Newton curvature model.

    The Hessian is assembled afresh at every iteration, so nothing is carried between
    them and the state is ``None``. What the model multiplies by is the assembled
    tensor, one block per leading axis of ``hessian_tag.batch_axes`` when the tag
    reports block structure.
    """

    hessian_fn: Callable[..., PyTree[Array]]

    def init(self, params: Y, *args: Any) -> None:
        del params, args
        return None

    def update(
        self,
        state: None,
        params: Y,
        y_diff: Y,
        grad: Y,
        *args: Any,
    ) -> tuple[HvpFn, PyTree[Array], None]:
        del state, grad, y_diff
        hessian_tensor = self.hessian_fn(params, *args)

        def _hvp(v: Y, tag: HessianTag | None) -> Y:
            return self.hvp(hessian_tensor, params, v, tag)

        return _hvp, hessian_tensor, None

    def hvp(
        self,
        hessian_repr: PyTree[Array],
        params: Y,
        v: Y,
        hessian_tag: HessianTag | None,
    ) -> Y:
        del params

        def apply_operator(H_b, v_b):
            return lx.PyTreeLinearOperator(H_b, jax.eval_shape(lambda: v_b)).mv(v_b)

        return map_blocks(apply_operator, hessian_repr, (v,), hessian_tag=hessian_tag)
