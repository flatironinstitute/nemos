"""Type variables and aliases shared across the second-order module.

They live here rather than in whichever file happened to need one first, so that a
module importing a type variable does not thereby import an implementation it has no
use for: before this, ``_direction`` imported ``_curvature`` for ``Y`` alone.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from ..._hess import HessianTag

# parameters
Y = TypeVar("Y")
# what a solver or a curvature model carries between iterations
S = TypeVar("S")
# how B_k is represented at this iterate, which is what ``hvp`` multiplies from: the
# pair history in the compact representation of Byrd et al., or the assembled Hessian.
# Neither is the curvature; both determine it.
R = TypeVar("R")
# what a direction carries between iterations
D = TypeVar("D")
# whatever the objective returns alongside its value
Aux = TypeVar("Aux")

# Multiply by the curvature model: ``(v, tag) -> B_k v``. The tag is the direction's,
# and says how the parameters are blocked; ``None`` when nothing is blocked.
HvpFn = Callable[[Y, "HessianTag | None"], Y]
