"""The penalized quadratic subproblem shared by NeMoS' proximal second-order solvers."""

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic

import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array, Bool, Scalar
from optimistix._misc import cauchy_termination

from .. import tree_utils
from ..typing import Params
from ._fista import FISTA
from ._line_search_mixins import LineSearchState, Y

if TYPE_CHECKING:
    from ..regularizer import Regularizer


class CompositeQuadraticMixin(Generic[Y]):
    r"""Minimize :math:`f(\beta) + P(\beta)` by repeatedly solving a quadratic model of :math:`f`.

    Each iteration solves

    .. math::
        \min_d \; \nabla f^\top d + \tfrac{1}{2} d^\top H d + P(\beta + d)

    with :class:`~nemos.solvers._fista.FISTA`, then backtracks on the composite
    objective. This is the proximal Newton-type scheme of Lee, Sun & Saunders [1]_; the
    curvature model :math:`H` is left to the host, which supplies it through
    ``_curvature`` and applies it through ``_hvp``. An assembled Hessian gives
    :class:`~nemos.solvers._newton.ProximalNewton`, a limited-memory approximation gives
    :class:`~nemos.solvers._lbfgs.ProximalLBFGS`.

    Well-posedness of the subproblem needs two conditions:

    - :math:`H \succeq 0`, making it convex. Definiteness is only needed to invert
      :math:`H`, and here :math:`H` is only multiplied.
    - :math:`\nabla f` restricted to :math:`\ker H` dominated by the growth of :math:`P`,
      making it bounded below. This constrains the quadratic model at the current iterate,
      not :math:`f`: a loss bounded below still has an unbounded model wherever :math:`H`
      is singular and the gradient has a component in :math:`\ker H`.

    A singular :math:`H` is therefore not by itself a problem, and no :math:`\ell_2` term
    is needed to supply the missing curvature. An indefinite :math:`H` is unsupported: the
    subproblem is unbounded below, so no solver has a minimum to find.

    Hosts must provide ``_hvp(grad, H, d)``. This mixin deliberately does not define it,
    not even as a raising stub: it precedes ``HessianMixin`` in the MRO of
    :class:`~nemos.solvers._newton.ProximalNewton`, so a stub here would shadow the
    assembled-Hessian implementation.

    References
    ----------
    .. [1] Lee, J. D., Sun, Y., & Saunders, M. A. (2014).
        "Proximal Newton-type methods for minimizing composite functions."
        *SIAM Journal on Optimization*, 24(3), 1420-1443.
        https://doi.org/10.1137/130921428
    .. [2] Tseng, P., & Yun, S. (2009).
        "A coordinate gradient descent method for nonsmooth separable minimization."
        *Mathematical Programming*, 117(1-2), 387-423.
        https://doi.org/10.1007/s10107-007-0170-0
    """

    # The smooth objective is the unregularized loss; the penalty is reached through a
    # proximal operator. ``HessianMixin.setup_hessian`` reads this to decide whether the
    # penalty's curvature belongs in the Hessian.
    _proximal: ClassVar[bool] = True

    def _resolve_loss(
        self,
        unregularized_loss: Callable,
        regularizer: "Regularizer",
        regularizer_strength: float | None,
        init_params: Params,
    ) -> Callable:
        """The unregularized loss, plus the penalty accessors the composite pieces need.

        ``self.fun`` ends up being the smooth part alone, so the composite objective is
        ``self.fun + self._penalty`` -- exactly what ``regularizer.penalized_loss`` builds
        from the same accessor.
        """
        self.prox = regularizer.get_proximal_operator(
            params=init_params, strength=regularizer_strength
        )
        self._penalty = regularizer.penalty_fn(
            params=init_params, strength=regularizer_strength
        )
        return unregularized_loss

    def _init_composite(
        self,
        inner_iter: int,
        inner_atol: float,
        inner_rtol: float,
    ) -> None:
        """Build the subproblem solver. Call after ``_init_loop``, which sets ``prox``."""
        self.inner_iter = inner_iter
        self.inner_atol = inner_atol
        self.inner_rtol = inner_rtol

        # The subproblem is solved for the new parameters, so the prox is the
        # regularizer's own and the solver does not depend on the current iterate:
        # build it once rather than per outer iteration.
        self._inner_solver = FISTA(
            atol=inner_atol,
            rtol=inner_rtol,
            norm=lx.internal.two_norm,
            prox=self.prox,
            while_loop_kind="lax",
        )

    def _direction(self, grad: Y, H: Any, params: Y) -> Y:
        r"""Minimize :math:`\nabla f^\top (z - \beta) + \frac12 (z - \beta)^\top H (z - \beta) + P(z)`.

        Solving for the new parameters :math:`z` rather than the step keeps the penalty
        where it is defined, so ``self.prox`` applies unchanged and the inner solver does
        not depend on the current iterate.

        The proximal operator carries metadata defined on the whole parameter tree --
        ``GroupLasso``'s mask, or a per-feature strength -- so the subproblem is solved
        on the full tree and only the Hessian-vector product is split per block. That
        keeps every regularizer usable without slicing each one's penalty metadata.
        """

        def quadratic(z, _):
            step = tree_utils.tree_sub(z, params)
            hvp = self._hvp(grad, H, step)
            return lx.internal.tree_dot(grad, step) + 0.5 * lx.internal.tree_dot(
                step, hvp
            )

        new_params = optx.minimise(
            quadratic,
            self._inner_solver,
            y0=params,
            max_steps=self.inner_iter,
            throw=False,
        ).value
        # ``_apply_or_reject`` scales and adds the result, so return the step
        return tree_utils.tree_sub(new_params, params)

    def _converged(
        self, params: Y, state: LineSearchState[Y], grad: Y, fval: Scalar
    ) -> Bool[Array, ""]:
        """Cauchy criterion on the accepted step, as :class:`~nemos.solvers._fista.FISTA` uses.

        A gradient-based test is unusable here: this solver differentiates the smooth
        part only, so its gradient does not vanish at the optimum of a composite
        objective, and any residual built from it inherits the curvature scale -- on
        badly conditioned data it never falls below ``tol`` even once the iterate has
        stopped moving.
        """
        del grad
        return cauchy_termination(
            self.rtol,
            self.tol,
            lx.internal.two_norm,
            params,
            state.y_diff,
            fval,
            fval - state.stats.function_val,
        )

    def _line_search_inputs(
        self, params: Y, step: Y, grad: Y, fval: Scalar, *args: Any
    ) -> tuple[Scalar, Y, Callable[[Y], Scalar]]:
        r"""Feed the composite objective and its slope to the inherited line search.

        Tseng & Yun (2009) [2]_ require the sufficient-decrease slope of a composite
        objective to be

        .. math::
            \Delta = \nabla f^\top d + P(\beta + d) - P(\beta),

        the :math:`P` difference being what makes :math:`\Delta < 0` a descent
        certificate when :math:`F` is nonsmooth. Since the search only ever forms
        ``vdot(step, slope)``, adding the penalty difference along ``step`` reproduces
        :math:`\Delta` exactly, and the stock Armijo search then applies unchanged.
        """
        penalty = self._penalty(params)
        penalty_diff = self._penalty(tree_utils.tree_add(params, step)) - penalty
        sq_norm = lx.internal.tree_dot(step, step)
        slope = tree_utils.tree_add_scalar_mul(
            grad, jnp.where(sq_norm > 0.0, penalty_diff / sq_norm, 0.0), step
        )
        return (
            fval + penalty,
            slope,
            lambda p: self.fun(p, *args) + self._penalty(p),
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return super().get_accepted_arguments() | {
            "inner_iter",
            "inner_atol",
            "inner_rtol",
        }
