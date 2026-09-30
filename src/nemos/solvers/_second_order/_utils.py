from typing import Any, Callable

import jax
from jaxtyping import Array, PyTree

from ..._hess import HessianTag, MatrixStructure


def map_blocks(
    fn: Callable[..., Any],
    H: PyTree[Array],
    trees: tuple[Any, ...],
    hessian_tag: HessianTag | None,
    block_state: PyTree[Array] | None = None,
) -> Any:
    """Run ``fn(H_blk, *tree_blks, state_blk)`` once per block and restack.

    ``trees`` are parameter-shaped and map along ``batch_axes``; ``block_state``
    maps along 0 and comes back along 0. Without a block-diagonal tag there is one
    block, so ``fn`` is applied once to the whole of each argument.
    """
    inps = (H, *trees) if block_state is None else (H, *trees, block_state)
    if (
        hessian_tag is None
        or hessian_tag.structure is not MatrixStructure.BLOCK_DIAGONAL
    ):
        return fn(*inps)

    axes = hessian_tag.batch_axes
    batch_axes = [0] + [axes for _ in range(len(trees))]
    out_axes = axes
    if block_state is not None:
        batch_axes.append(0)
        out_axes = (out_axes, 0)
    return jax.vmap(fn, in_axes=batch_axes, out_axes=out_axes)(*inps)
