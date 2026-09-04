"""Abstract class for regression models."""

from __future__ import annotations

import abc
import warnings
from abc import abstractmethod
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Tuple, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import solvers, tree_utils, utils
from ._hess import (
    HessianTag,
    LeafClaim,
    MatrixProperty,
    MatrixStructure,
    claim_nothing,
    mask_of_claim,
)
from ._regularizer_builder import AVAILABLE_REGULARIZERS, instantiate_regularizer
from .base_class import Base
from .params import ModelParams
from .pytrees import FeaturePytree
from .regularizer import GroupLasso, Regularizer
from .solvers import SolverProtocol, SolverSpec
from .solvers._newton import Newton
from .solvers._no_op import NoOpSolver
from .type_casting import cast_to_jax, is_numpy_array_like
from .typing import (
    DESIGN_INPUT_TYPE,
    ModelParamsT,
    SolverInit,
    SolverRun,
    SolverState,
    SolverUpdate,
    StepResult,
    UserProvidedParamsT,
    ValidatorT,
)
from .utils import _flatten_dict, _get_name, _unpack_params, get_env_metadata

_SOLVER_ARGS_CACHE = {}


def strip_metadata(arg_num: Optional[int] = None, arg_name: Optional[str] = None):
    """Strip metadata from arg."""
    if arg_num is None or arg_name is None:
        raise ValueError("Must specify either arg_num or kwarg_key.")

    def decorator(func):
        """Strip metadata if available."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if arg_name in kwargs:
                inp = kwargs[arg_name]
            else:
                inp = args[min(arg_num, len(args) - 1)]
            self._metadata = {
                "metadata": inp._metadata if hasattr(inp, "_metadata") else None,
                "columns": inp.columns if hasattr(inp, "columns") else None,
            }
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class BaseRegressor(
    abc.ABC, Base, Generic[UserProvidedParamsT, ModelParamsT, ValidatorT]
):
    """Abstract base class for GLM regression models.

    This class encapsulates the common functionality for Generalized Linear Models (GLM)
    regression models. It provides an abstraction for fitting the model, making predictions,
    scoring the model, simulating responses, and preprocessing data. Concrete classes
    are expected to provide specific implementations of the abstract methods defined here.
    Below is a table listing the default and available solvers for each regularizer.

    | Regularizer   | Default Solver   | Available Solvers                                           |
    | ------------- | ---------------- | ----------------------------------------------------------- |
    | UnRegularized | LBFGS            | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    | Ridge         | LBFGS            | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    | Lasso         | ProximalGradient | ProximalGradient                                            |
    | GroupLasso    | ProximalGradient | ProximalGradient                                            |

    Parameters
    ----------
    regularizer :
        Regularization to use for model optimization. Defines the regularization scheme
        and related parameters.
        Default is UnRegularized regression.
    regularizer_strength :
        Float that is default None. Sets the regularizer strength. If a user does not pass a value, and it is needed for
        regularization, a warning will be raised and the strength will default to 1.0.
    solver_name :
        Solver to use for model optimization. Defines the optimization scheme and related parameters.
        The solver must be an appropriate match for the chosen regularizer.
        Default is `None`. If no solver specified, one will be chosen based on the regularizer.
        Please see table above for regularizer/optimizer pairings.
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g. stepsize, tol, acceleration, etc.
         For details on each solver's kwargs, see `get_accepted_arguments` and `get_solver_documentation`.

    See Also
    --------
    Concrete models:

    - [`GLM`](../glm/#nemos.glm.GLM): A feed-forward GLM implementation.
    - [`PopulationGLM`](../glm/#nemos.glm.PopulationGLM): A population GLM implementation.
    """

    _validator: ValidatorT

    # Sparsity of the loss Hessian, and which axis of each parameter is the batch when it
    # is block diagonal. ``MatrixStructure.FULL`` claims nothing: no sparsity to exploit.
    _hess_structure: MatrixStructure = MatrixStructure.FULL
    _hess_batch_axes: Any = None

    # overwrite this in subclasses if their objective functions return aux
    _has_aux: bool = False

    # user setting: fixed-parameter spec (array leaf = fixed value, None leaf = learn).
    _fix_params: Optional[ModelParamsT] = None

    def __init__(
        self,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Any = None,
        solver_name: Optional[str] = None,
        solver_kwargs: Optional[dict] = None,
    ):
        self._solver_spec = None
        self.regularizer = "UnRegularized" if regularizer is None else regularizer
        self.regularizer_strength = regularizer_strength

        self.solver_name = solver_name

        if solver_kwargs is None:
            solver_kwargs = dict()

        solver_class = self.solver_spec.implementation
        self._check_solver_kwargs(solver_class, solver_kwargs)

        self.solver_kwargs = solver_kwargs
        self._optimizer_init_state = None
        self._optimizer_update = None
        self._optimizer_run = None

    def __sklearn_tags__(self):
        """Return regression model specific estimator tags."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.non_deterministic = True
        tags.requires_fit = True
        # conversion happens internally
        tags.array_api_support = True
        return tags

    @property
    def optimizer_init_state(self) -> Union[None, SolverInit]:
        """
        Provides the initialization function for the optimizer state.

        This function is responsible for initializing the optimizer state, necessary for the start
        of the optimizer process. It sets up initial values for parameters like gradients and step
        sizes based on the model configuration and input data.

        Returns
        -------
        :
            The function to initialize the optimizer state, if available; otherwise, None if
            the optimizer has not yet been instantiated.
        """
        return self._optimizer_init_state

    @property
    def optimizer_update(self) -> Union[None, SolverUpdate]:
        """
        Provides the function for updating the state during the optimization process.

        This function is used to perform a single update step in the optimization process. It updates
        the model's parameters based on the current state, data, and gradients. It is typically used
        in scenarios where fine-grained control over each optimizer step is necessary, such as in
        online learning or complex optimization scenarios.

        Returns
        -------
        :
            The function to perform a single optimization update step, if available; otherwise, None if
            the optimizer has not yet been instantiated.
        """
        return self._optimizer_update

    @property
    def optimizer_run(self) -> Union[None, SolverRun]:
        """
        Provides the function to execute the optimization process.

        This function runs the optimizer using the initialized parameters and state, performing the
        optimization to fit the model to the data. It iteratively updates the model parameters until
        a stopping criterion is met, such as convergence or exceeding a maximum number of iterations.

        Returns
        -------
        :
            The function to run the optimization process, if available; otherwise, None if
            the optimizer has not yet been instantiated.
        """
        return self._optimizer_run

    def set_params(self, **params: Any):
        """Manage warnings in case of multiple parameter settings."""
        if "regularizer" in params:
            # override _regularizer_strength to None to avoid conficts between regularizers
            self._regularizer_strength = None

            if "regularizer_strength" in params:
                # if both regularizer and regularizer_strength are set, then only
                # warn in case the strength is not expected for the regularizer type
                reg = params.pop("regularizer")
                super().set_params(regularizer=reg)

            elif self.regularizer_strength is not None:
                reg = params.pop("regularizer")
                super().set_params(regularizer=reg)

        return super().set_params(**params)

    @property
    def regularizer(self) -> Union[None, Regularizer]:
        """Getter for the regularizer attribute."""
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer: Union[str, Regularizer]):
        """Setter for the regularizer attribute."""
        # instantiate regularizer if str
        if isinstance(regularizer, str):
            self._regularizer = instantiate_regularizer(name=regularizer)
        elif isinstance(regularizer, Regularizer):
            self._regularizer = regularizer
        else:
            raise TypeError(
                f"The regularizer should be either a string from "
                f"{AVAILABLE_REGULARIZERS} or an instance of `nemos.regularizer.Regularizer`"
            )

        # force check of regularizer_strength
        # need to use hasattr to avoid class instantiation issues
        if hasattr(self, "_regularizer_strength"):
            self.regularizer_strength = self._regularizer_strength

        # check if solver is not allowed, if it isn't revert to default.
        # note that, if self._solver_spec is None (default) -> solver always
        # allowed, so no warning.
        if self.solver_name not in self.regularizer.allowed_solvers:
            warnings.warn(
                f"Solver ``{self.solver_name}`` is not allowed for regularizer {self._regularizer}. "
                f"Overriding solver with the default allowed solver {self._regularizer.default_solver}.",
                UserWarning,
                stacklevel=2,
            )
            self.solver_name = None
        else:
            self._invalidate_solver()

    @property
    def regularizer_strength(self) -> Any:
        """Regularizer strength getter."""
        return self._regularizer_strength

    @regularizer_strength.setter
    def regularizer_strength(self, strength: Any):
        self._regularizer_strength = self.regularizer._validate_strength(strength)
        self._invalidate_solver()

    @property
    def solver_name(self) -> str:
        """Getter for the solver_name attribute."""
        return self.solver_spec.algo_name

    @solver_name.setter
    def solver_name(self, solver_name: str | None):
        """Setter for the solver_name attribute."""
        if not isinstance(solver_name, str) and solver_name is not None:
            raise TypeError("solver_name must be a string.")
        elif solver_name is None:
            self._solver_spec = None
        else:
            # check if solver str passed is valid for regularizer
            spec = solvers.get_solver(solver_name)
            self._regularizer.check_solver(spec.algo_name)
            self._solver_spec = spec
        self._invalidate_solver()

    def _hess_leaf_claims(
        self, params: ModelParamsT, active_spec: ModelParams[bool]
    ) -> ModelParams[LeafClaim]:
        """Say what the loss Hessian certifies about each leaf's own block.

        A claim here is about the loss alone. It has to hold at every parameter value, not
        only at the optimum, and it has to be justified without inspecting the data — the
        regularizer's claims are added later, by ``combine_hessian_tags``.

        The default certifies nothing, so a subclass that does not override this still gets
        a correct tag. A subclass labels a leaf ``LeafClaim.DEFINITE`` only when its
        block is definite for a reason that survives any design matrix: anything that comes
        down to the rank of ``X`` costs the factorization the tag exists to avoid. It labels
        a leaf ``LeafClaim.FLAT`` only when the loss has no curvature there at all,
        which for a likelihood means it does not use that parameter.

        Parameters
        ----------
        params :
            The parameters being fitted, i.e. the ones left after the fixed ones are
            partitioned out.
        active_spec :
            The filter spec ``params`` was partitioned with, as returned by
            ``_active_filter_spec``. It is the one unambiguous statement of which leaves
            are being fitted: a ``None`` leaf in ``params`` means "frozen" in the active
            half of the partition and "fitted" in the frozen half, so the tree alone cannot
            be asked.

        Returns
        -------
        :
            A tree shaped like ``params`` carrying one :class:`~nemos._hess.LeafClaim`
            member per leaf: ``LeafClaim.FLAT``, ``LeafClaim.DEFINITE`` or
            ``LeafClaim.UNCLAIMED``.
        """
        return claim_nothing(params)

    def _resolve_hess_property(self) -> MatrixProperty:
        """Give the sign the loss Hessian has at every parameter value.

        The default certifies nothing: an arbitrary loss has an arbitrary Hessian, and a
        sign claimed here that the matrix does not have sends Newton into a Cholesky
        factorization of a matrix it cannot factor. A subclass returns something stronger
        when the shape of its loss says so, e.g. a GLM whose inverse link keeps the
        likelihood convex has a positive semidefinite Hessian everywhere.
        """
        return MatrixProperty.SYMMETRIC

    def _resolve_hess_tag(self, params: ModelParamsT) -> HessianTag:
        """Describe the loss Hessian at these parameters, leaving the penalty aside.

        The tag is assembled from three overridable pieces, each defaulting to a claim of
        nothing, so a new model gets a usable tag without implementing anything:

        - ``_hess_structure`` and ``_hess_batch_axes``: the sparsity, ``MatrixStructure.FULL`` by default,
          and which axis of each parameter is the batch when it is block diagonal.
        - ``_resolve_hess_property()``: the sign, ``MatrixProperty.SYMMETRIC`` by default.
        - ``_hess_leaf_claims(params, active_spec)``: what is certified about each leaf's
          own block, nothing by default.

        It is built against the parameters being fitted rather than declared on the class,
        because which parameters those are depends on what is held fixed, and a claim about
        a parameter that is not being fitted describes a block of a matrix that does not
        exist.

        Parameters
        ----------
        params :
            The parameters being fitted.

        Returns
        -------
        :
            The tag for the loss Hessian, with the two leaf sets read off the per-leaf
            claims. ``Newton`` combines it with the regularizer's tag to pick a linear
            solver; see :func:`~nemos._hess.combine_hessian_tags`.
        """
        claims = self._hess_leaf_claims(params, self._active_filter_spec())
        return HessianTag(
            structure=self._hess_structure,
            property=self._resolve_hess_property(),
            batch_axes=self._hess_batch_axes,
            flat_on=mask_of_claim(claims, LeafClaim.FLAT),
            definite_on=mask_of_claim(claims, LeafClaim.DEFINITE),
        )

    def _resolve_default_solver(self) -> str:
        """Name of the default solver when the user has not set one.

        Defaults to the regularizer's own default solver. Subclasses may override to
        express a model- and regularizer-specific preference (e.g. GLMs default to
        Newton when the regularizer makes the Hessian positive definite).
        """
        return self.regularizer.default_solver

    @property
    def solver_spec(self) -> SolverSpec:
        """Getter for the solver specification."""
        if self._solver_spec is None:
            return solvers.get_solver(self._resolve_default_solver())
        return self._solver_spec

    @property
    def solver_kwargs(self):
        """Getter for the solver_kwargs attribute."""
        return self._solver_kwargs

    @solver_kwargs.setter
    def solver_kwargs(self, solver_kwargs: dict):
        """Setter for the solver_kwargs attribute."""
        if solver_kwargs:
            solver_cls = self.solver_spec.implementation
            self._check_solver_kwargs(solver_cls, solver_kwargs)
        self._solver_kwargs = solver_kwargs
        self._invalidate_solver()

    @staticmethod
    def _check_solver_kwargs(solver_class: Type, solver_kwargs: dict[str, Any]) -> None:
        """
        Check if provided solver keyword arguments are valid.

        Parameters
        ----------
        solver_class :
            Class of the solver.
        solver_kwargs :
            Additional keyword arguments for the solver.

        Raises
        ------
        NameError
            If any of the solver keyword arguments are not valid.
        """
        accepted_args = solver_class.get_accepted_arguments()

        undefined_kwargs = set(solver_kwargs.keys()) - set(accepted_args)

        if undefined_kwargs:
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for {solver_class.__name__}!"
            )

    def _invalidate_solver(self):
        self._solver = None
        self._solver_loss_fun = None
        self._optimizer_init_state = None
        self._optimizer_update = None
        self._optimizer_run = None

    def _no_op_optimizer(self) -> SolverState:
        """Install :class:`NoOpSolver`, for when the active parameter tree is empty."""
        warnings.warn(
            "Every parameter is fixed, through `fix_params` and/or `fit_intercept=False`; "
            "no optimization will run and the fixed values are returned unchanged.",
            UserWarning,
        )
        self._solver = NoOpSolver()
        self._optimizer_init_state = self._solver.init_state
        self._optimizer_update = self._solver.update
        self._optimizer_run = self._solver.run
        return self._optimizer_init_state(None)

    def _partition_active(
        self, params: ModelParamsT
    ) -> Tuple[ModelParamsT, ModelParamsT]:
        """
        Compute active and frozen parameter trees.

        Parameters
        ----------
        params:
            The model parameters.

        Returns
        -------
        :
            A tuple containing the active and frozen parameter trees.

        """
        return eqx.partition(params, self._active_filter_spec())

    def _active_filter_spec(self) -> ModelParams[bool]:
        """Boolean filter spec (tree-prefix) marking the actively optimized leaves.

        Derived from ``_fix_params`` alone: a leaf is active iff the spec holds
        ``None`` there. Subclasses fold model-specific settings in (e.g. the GLM
        freezes the intercept when ``fit_intercept=False``).
        """
        return jax.tree_util.tree_map(
            lambda x: x is None, self._fix_params, is_leaf=lambda x: x is None
        )

    def _frozen_values(
        self, X: DESIGN_INPUT_TYPE, y: jnp.ndarray
    ) -> Optional[ModelParamsT]:
        """Values the frozen leaves are pinned to, implied by the model settings.

        Complement of :meth:`_active_filter_spec`: the spec marks *which* leaves are
        actively optimized, this returns *what* the remaining leaves are held at
        (tree-prefix with ``None`` on active leaves; ``None`` when nothing is frozen).
        Derived from ``_fix_params`` alone here — its array leaves are the fixed
        values. Subclasses fold model-specific settings in (e.g. the GLM pins the
        intercept at zero when ``fit_intercept=False``); ``X`` and ``y`` let them
        infer the shape of a pinned leaf.
        """
        return self._fix_params

    def _normalize_user_params(
        self,
        init_params: UserProvidedParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> UserProvidedParamsT:
        """Complete a user-provided parameter set before validation.

        User-facing entry points (``fit``, ``initialize_optimizer_and_state``) accept
        parameters in a convenient, possibly *incomplete* form: leaves that the model
        will not learn may be omitted (passed as ``None``) so the user does not have to
        supply a value for something that is held fixed. This hook is the single seam
        where such input is turned into a complete, concrete parameter set, filling in
        the omitted leaves with their fixed defaults (and warning if the user supplied a
        value for a leaf that will not be estimated).

        It is separate from the other parameter-processing steps because it is the only
        one allowed to *change values*, and the only one that is inherently model
        specific:

        - ``_normalize_user_params`` (this method): inject defaults for omitted/fixed
          leaves, coerce/override, warn. Model specific — the base class does not know
          which leaves a subclass can leave unset (e.g. the GLM intercept when
          ``fit_intercept=False``), so it is a no-op here and subclasses override it.
        - ``validate_and_cast_params``: validate structure/dtype/shape and cast the
          user tuple to a ``ModelParams`` pytree. Assumes a *complete* set of concrete
          arrays, which is why normalization must run first.
        - ``validate_consistency``: check the parameters against the data (feature and
          output dimensions).
        - ``_partition_active``: split the concrete parameters into the active subtree
          the solver optimizes and the frozen subtree recombined afterwards. The fixed
          values injected here are what end up in the frozen subtree.

        Running order is therefore: normalize -> validate/cast -> check consistency ->
        partition. The default implementation returns ``init_params`` unchanged.

        Parameters
        ----------
        init_params :
            User-provided parameters, in the model's user-facing format.
        X :
            Input predictors, used to infer the shape/default of any omitted leaf.
        y :
            Target data, used to infer the shape/default of any omitted leaf.

        Returns
        -------
        :
            The parameters with any omitted fixed leaves filled in.
        """
        return init_params

    def _instantiate_solver(
        self,
        loss,
        init_params: ModelParamsT,
        solver_name: Optional[str] = None,
        solver_kwargs: Optional[dict] = None,
        regularizer: Optional[Regularizer] = None,
        regularizer_strength: Optional[Any] = None,
        frozen_params: Optional[ModelParamsT] = None,
    ) -> SolverProtocol:
        """
        Instantiate the solver with the provided loss function.

        Instantiate the solver with the provided loss function, and store callable functions
        that initialize the solver state, update the model parameters, and run the optimization
        as attributes.

        This method creates a solver instance from the solver registry, tailored to
        the specific loss function and regularization approach defined by the Regularizer instance.
        It also handles the proximal operator if required for the optimization method. The returned
        functions are directly usable in optimization loops, simplifying the syntax by pre-setting
        common arguments like regularization strength and other hyperparameters.

        Solvers are expected to adhere to the `AbstractSolver` API.

        Parameters
        ----------
        loss:
            The un-regularized loss function.
        init_params:
            The model parameters.
        solver_name:
            Optional solver name, default is self.solver_name.
        solver_kwargs:
            Optional dictionary with the solver kwargs.
            If nothing is provided, it defaults to self.solver_kwargs.
        regularizer:
            Optional regularizer, default is self.regularizer.
        regularizer_strength:
            Optional regularization strength, default is self.regularizer_strength.
        frozen_params:
            Set of fixed parameters that will be combined with actively learned ``init_params``.

        Returns
        -------
        :
            The solver instance.
        """
        # final check that solver is valid for chosen regularizer
        self._regularizer.check_solver(self.solver_spec.algo_name)

        if solver_kwargs is None:
            # copy dictionary of kwargs to avoid modifying user settings
            solver_kwargs = deepcopy(self.solver_kwargs)
        if solver_name is None:
            solver_name = self.solver_spec.full_name
        if regularizer is None:
            regularizer = self.regularizer
        if regularizer_strength is None:
            regularizer_strength = self.regularizer_strength

        # instantiate the solver
        solver_cls = solvers.get_solver(solver_name).implementation

        self._check_solver_kwargs(solver_cls, solver_kwargs)

        if frozen_params is not None:

            def _loss(params, *args, **kwargs):
                params = eqx.combine(params, frozen_params)
                return loss(params, *args, **kwargs)

        else:
            _loss = loss

        solver = solver_cls(
            _loss,
            regularizer,
            regularizer_strength,
            has_aux=self._has_aux,
            init_params=init_params,
            **solver_kwargs,
        )

        if isinstance(solver, Newton):
            solver.setup_hessian(
                self._get_hess_fn(frozen=frozen_params),
                self._resolve_hess_tag(init_params),
                regularizer._resolve_hess_tag(init_params, self.regularizer_strength),
            )

        # nemos's solvers store a .fun attribute, but it's not necessary for a solver to work.
        # A test relies on having _solver_loss_fun saved, so still check and save it if possible.
        # But it's not a problem if .fun doesn't exist in user-defined solvers.
        if hasattr(solver, "fun"):
            # check that the loss is Callable
            utils.assert_is_callable(solver.fun, "solver's loss")
            self._solver_loss_fun = solver.fun

        return solver

    def _get_hess_fn(self, frozen: Optional[ModelParamsT] = None) -> Optional[Callable]:
        return None

    @abc.abstractmethod
    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray],
        init_params: Optional[UserProvidedParamsT] = None,
    ) -> BaseRegressor[UserProvidedParamsT, ModelParamsT]:
        """Fit the model to neural activity."""
        pass

    @abc.abstractmethod
    def score(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray],
        # may include score_type or other additional model dependent kwargs
        **kwargs,
    ) -> jnp.ndarray:
        """Score the predicted firing rates (based on fit) to the target neural activity."""
        pass

    @abc.abstractmethod
    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
    ):
        """Simulate neural activity in response to a feed-forward input and recurrent activity."""
        pass

    @abc.abstractmethod
    def _get_model_params(self) -> ModelParamsT:
        """Pack coef_ and intercept_  into a params pytree."""
        pass

    @abc.abstractmethod
    def _set_model_params(self, params: ModelParamsT):
        """Unpack and store params pytree to coef_ and intercept_."""
        pass

    def get_model_params(self) -> UserProvidedParamsT:
        """
        Return the fitted model parameters in user-facing form.

        The exact structure depends on the concrete subclass (e.g.
        ``(coef, intercept)`` for a GLM), matching what
        :meth:`initialize_params` returns.

        Returns
        -------
        :
            The fitted parameters in user-facing form.
        """
        params = self._validator.from_model_params(self._get_model_params())
        # Make a kind of copy by rebuilding the pytree structure so callers
        # cannot mutate container-like model params (for example dict coefficients)
        # by changing the return value. This is fine for the current JAX-array leaves,
        # but it would need revisiting if future subclasses store mutable objects at the leaves.
        return jax.tree_util.tree_map(lambda x: x, params)

    @abc.abstractmethod
    def _compute_loss(
        self,
        params: ModelParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ):
        """Unpenalized scalar loss given parameters and data.

        For GLM-family models this is the negative log-likelihood passed to
        gradient-based solvers (the solver adds the regularization penalty on
        top). For HMM-family models the EM solver does not consume this method,
        but it is still implemented as the negative marginal log-likelihood so
        that ``score`` and ``compute_loss`` work uniformly across the hierarchy.

        Parameters
        ----------
        params :
            Model parameters.
        X :
            Predictors.
        y :
            Target neural activity.
        *args :
            Additional positional arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        :
            The unpenalized loss value (a scalar).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `_compute_loss`."
        )

    @cast_to_jax
    def compute_loss(
        self,
        params: UserProvidedParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute the loss function for the model.

        This method validates inputs and converts user-provided parameters to the internal
        representation before computing the loss.

        Parameters
        ----------
        params
            Parameter tuple of (coefficients, intercept).
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target data, array of shape ``(n_time_bins,)`` for single neuron models or
            ``(n_time_bins, n_neurons)`` for population models.
        *args
            Additional positional arguments passed to the model-specific loss function.
        **kwargs
            Additional keyword arguments passed to the model-specific loss function.

        Returns
        -------
        loss
            The loss value (negative log-likelihood).

        Raises
        ------
        ValueError
            If inputs or parameters have incompatible shapes or invalid values.
        """
        self._validator.validate_inputs(X, y)
        params = self._validator.validate_and_cast_params(params)
        self._validator.validate_consistency(params, X, y)
        X, y = self._preprocess_inputs(X, y)
        return self._compute_loss(params, X, y, *args, **kwargs)

    @abc.abstractmethod
    def update(
        self,
        params: UserProvidedParamsT,
        opt_state: SolverState,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> StepResult:
        """Run a single update step of the underlying solver."""
        pass

    @cast_to_jax
    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> UserProvidedParamsT:
        """Initialize model parameters.

        Initialize coefficients with zeros and intercept by matching the mean firing rate.

        Parameters
        ----------
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target data, array of shape ``(n_time_bins,)`` for single neuron models or
            ``(n_time_bins, n_neurons)`` for population models.

        Returns
        -------
        params
            Initial parameter tuple of (coefficients, intercept).
        """
        init_params = self._model_specific_initialization(X, y)
        return self._validator.from_model_params(init_params)

    @abc.abstractmethod
    def _model_specific_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        **kwargs,
    ) -> ModelParamsT:
        """Model specific initialization logic."""
        pass

    def _preprocess_inputs(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Optional[jnp.ndarray] = None,
        *args: jnp.ndarray,
        drop_nans: bool = True,
    ) -> Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray, ...] | None:
        """Preprocess inputs before initializing state."""
        X, y = cast_to_jax(lambda *x: x)(X, y)
        if drop_nans:
            res = tree_utils.drop_nans(X, y, *args)
            X, y = res[:2]
            args = res[2:]

        data = X.data if isinstance(X, FeaturePytree) else X

        if isinstance(self.regularizer, GroupLasso):
            if self.regularizer.mask is None and is_numpy_array_like(data)[1]:
                warnings.warn(
                    "Mask has not been set. Defaulting to a single group for all parameters. "
                    "Please see the documentation on GroupLasso regularization for defining a mask."
                )
            elif self.regularizer.mask is not None:
                self._wrap_grouplasso_mask(data, y)

        return data, y, *args

    def _wrap_grouplasso_mask(
        self,
        data: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> None:
        """Convert a user-provided GroupLasso mask into the internal structured format.

        Mutates ``self.regularizer.mask`` in place. No-op if the mask is already
        in the structured format (i.e. already an instance of the solved params
        type). For composite models (e.g. GLM-HMM) the numerical solver optimizes
        only a sub-pytree of the full parameters; the mask is interpreted at that
        level.
        """
        import equinox as eqx

        model_pars = self._validator.get_empty_params(data, y)
        # composite models solve only a sub-pytree; flat models solve the full
        # params. The regularizer and its mask act at the solved level.
        solver_subtree = getattr(
            model_pars, "solver_param_subtree", lambda: lambda p: p
        )()
        solver_pars = solver_subtree(model_pars)
        if isinstance(self.regularizer.mask, type(solver_pars)):
            return

        select_subtrees = (
            solver_pars.regularizable_subtrees()
            if hasattr(solver_pars, "regularizable_subtrees")
            else [lambda p: p]
        )
        if len(select_subtrees) == 1:
            mask_list = [self.regularizer.mask]
        else:
            mask_list = jax.tree_util.tree_leaves(self.regularizer.mask)
            if len(mask_list) != len(select_subtrees):
                raise ValueError(
                    f"{type(self).__name__} has {len(select_subtrees)} regularizable "
                    f"parameters but the mask pytree has {len(mask_list)} leaves; "
                    f"provide a pytree with one leaf per regularizable parameter."
                )

        for where, m in zip(select_subtrees, mask_list):
            expected = jax.tree_util.tree_structure(where(solver_pars))
            actual = jax.tree_util.tree_structure(m)
            if expected != actual:
                raise ValueError(
                    f"Mask pytree structure {actual} does not match the expected "
                    f"parameter structure {expected}. The mask must mirror the "
                    f"structure of the corresponding parameter (e.g. if X is a "
                    f"list, the mask must also be a list)."
                )

        struct = jax.tree_util.tree_structure(solver_pars)
        mask_tree = jax.tree_util.tree_unflatten(struct, [None] * struct.num_leaves)
        for where, m in zip(select_subtrees, mask_list):
            mask_tree = eqx.tree_at(where, mask_tree, m, is_leaf=lambda x: x is None)
        self.regularizer.mask = mask_tree

    @abc.abstractmethod
    def _initialize_optimizer_and_state(
        self,
        init_params: ModelParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        frozen_params: Optional[ModelParamsT] = None,
    ) -> SolverState:
        """Initialize the optimizer and the state of the optimizer for running fit and update."""
        pass

    @cast_to_jax
    def initialize_optimizer_and_state(
        self,
        init_params: UserProvidedParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        **kwargs,
    ) -> SolverState:
        """Initialize the optimization routine and its state for running fit and update.

        This method must be called before using :meth:`update` for iterative optimization.
        It sets up the solver with the provided initial parameters and data.

        Parameters
        ----------
        init_params :
            Initial parameter tuple of (coefficients, intercept).
        X :
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y :
            Target data, array of shape ``(n_time_bins,)`` for single neuron models or
            ``(n_time_bins, n_neurons)`` for population models.
        kwargs :
            Additional keyword arguments for validation.

        Returns
        -------
        state :
            Initial solver state.

        Raises
        ------
        ValueError
            If inputs or parameters have incompatible shapes or invalid values.
        """
        self._validator.validate_inputs(X=X, y=y, **kwargs)
        init_params = self._normalize_user_params(init_params, X, y)
        init_params = self._validator.validate_and_cast_params(init_params)
        self._validator.validate_consistency(init_params, X=X, y=y)
        X, y = self._preprocess_inputs(X, y, drop_nans=True)
        active, frozen = self._partition_active(init_params)
        state = self._initialize_optimizer_and_state(active, X, y, frozen_params=frozen)
        return state

    def _optimize_solver_params(self, X: DESIGN_INPUT_TYPE, y: jnp.ndarray) -> dict:
        """
        Compute and update solver parameters with optimal defaults if available.

        This method checks the current solver configuration and, if an optimal
        configuration is known for the given model parameters, computes the optimal
        batch size, step size, and other hyperparameters to ensure faster convergence.

        Parameters
        ----------
        X :
            Input data used to compute smoothness and strong convexity constants.
        y :
            Target values used in conjunction with X for the same purpose.

        Returns
        -------
        :
            A dictionary containing the solver parameters, updated with optimal defaults
            where applicable.

        """
        # Start with a copy of the existing solver parameters
        new_solver_kwargs = self.solver_kwargs.copy()

        # get the model specific configs
        (
            compute_defaults,
            compute_l_smooth,
            strong_convexity,
        ) = self._get_optimal_solver_params_config()
        if compute_defaults and compute_l_smooth:
            # Check if the user has provided batch size or stepsize, or else use None
            batch_size = new_solver_kwargs.get("batch_size", None)
            stepsize = new_solver_kwargs.get("stepsize", None)

            # Compute the optimal batch size and stepsize based on smoothness, strong convexity, etc.
            new_params = compute_defaults(
                compute_l_smooth,
                X,
                y,
                batch_size=batch_size,
                stepsize=stepsize,
                strong_convexity=strong_convexity,
            )

            # Update the solver parameters with the computed optimal values
            new_solver_kwargs.update(new_params)

        return new_solver_kwargs

    @abstractmethod
    def _get_optimal_solver_params_config(self):
        """Return the functions for computing default step and batch size for the solver."""
        pass

    @abstractmethod
    def save_params(
        self,
        filename: Union[str, Path],
    ):
        """Save model parameters and specified attributes to a .npz file."""
        pass

    def _save_params(
        self,
        filename: Union[str, Path],
        fit_attrs: dict,
        string_attrs: list = None,
    ):
        """
        Save model parameters and specified attributes to a .npz file.

        This is a private method intended to be used by subclasses to implement.
        Adds metadata about the jax and nemos versions used to save the model.

        Parameters
        ----------
        filename :
            The output filename.
        fit_attrs :
            Dictionary containing the fitting parameters specific to the subclass model.
        string_attrs :
            List of attributes to be saved as strings.
        """

        # extract model parameters
        model_params = self.get_params(deep=False)
        model_params = _unpack_params(model_params, string_attrs)

        # append the fit attributes to the model parameters
        model_params.update(fit_attrs)

        # set solver_kwargs to None so tha it can be saved in the npz
        if model_params["solver_kwargs"] == {}:
            model_params["solver_kwargs"] = None

        # save jax and nemos versions
        model_params["save_metadata"] = get_env_metadata()

        # save the model class name
        model_params["model_class"] = _get_name(self.__class__)

        # flatten the parameters dictionary to ensure it can be saved
        model_params = _flatten_dict(model_params)
        np.savez(filename, **model_params)

    def _get_fit_state(self) -> dict:
        """
        Collect all attributes that follow the fitted attribute convention.

        Collect all attributes ending with an underscore.

        Returns
        -------
        :
            A dictionary of attribute names and their values.
        """
        set_attr = {
            name: getattr(self, name)
            for name in dir(self)
            # sklearn has "_repr_html_" and "_repr_mimebundle_" methods
            # filter callables
            if name.endswith("_")
            and not name.endswith("__")
            and (not callable(getattr(self, name)))
        }
        # drop attributes that have a private equivalent
        # those are likely properties without a setter.
        private_set_attr_names = [
            name for name in set_attr.keys() if name.startswith("_")
        ]
        for name in private_set_attr_names:
            if name[1:] in set_attr:
                set_attr.pop(name[1:])
        return set_attr

    @staticmethod
    def _get_validator_extra_params() -> dict | None:
        """Get validator extra parameters.

        Provide instance specific validator configuration if needed.
        """
        return {}

    @staticmethod
    def _convergence_badge_html(solver_state) -> str:
        """Build the convergence diagnostic HTML for the model repr.

        Mirror the convergence detection used in ``GLM.fit``: prefer
        ``stats.converged``, fall back to a ``converged`` flag exposed directly
        by custom solvers, and otherwise report it as unknown. A missing solver
        state (e.g. for a model loaded from disk) is treated the same as a
        solver that does not report convergence.

        Parameters
        ----------
        solver_state :
            The model's ``solver_state_`` attribute, or ``None``.

        Returns
        -------
        str
            An HTML snippet displaying the convergence status.
        """
        if solver_state is None:
            converged = None
        elif hasattr(solver_state, "stats") and hasattr(
            solver_state.stats, "converged"
        ):
            converged = bool(solver_state.stats.converged)
        elif hasattr(solver_state, "converged"):
            converged = bool(solver_state.converged)
        else:
            converged = None

        if converged is None:
            c_color, c_text = ("#6c757d", "Unknown")
        elif converged:
            c_color, c_text = ("#28a745", "Yes")
        else:
            c_color, c_text = ("#dc3545", "No")
        return f'<span><strong>Converged:</strong> <span style="color: {c_color};">{c_text}</span></span>'

    def _repr_mimebundle_(self, **kwargs):
        """Mimebundle representation of the model.

        Wraps the default scikit-learn diagram with a small nemos diagnostics bar.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to the default scikit-learn mimebundle generator.

        Returns
        -------
        dict
            A dictionary mapping mime types to representation data.
        """
        bundle_func = getattr(super(), "_repr_mimebundle_", None)
        bundle = bundle_func(**kwargs) if bundle_func else {}

        if "text/html" not in bundle:
            html_func = getattr(super(), "_repr_html_", None)
            bundle["text/html"] = html_func() if html_func else repr(self)

        state = self._get_fit_state()
        coef = state.get("coef_")
        is_fitted = coef is not None

        state_color, state_text = (
            ("#28a745", "Fitted") if is_fitted else ("#dc3545", "Unfitted")
        )
        diagnostics = "</div>"

        if is_fitted:
            intercept_shape = getattr(state.get("intercept_"), "shape", ())
            n_neurons = (
                1
                if intercept_shape in ((), (1,))
                else getattr(intercept_shape, "__getitem__", lambda x: "N/A")(0)
            )

            def get_features(x):
                return getattr(x, "shape", (1,))[0] if getattr(x, "ndim", 0) > 0 else 1

            n_features = "Unknown"
            try:
                n_features = sum(
                    jax.tree_util.tree_flatten(
                        jax.tree_util.tree_map(get_features, coef)
                    )[0]
                )
            except Exception:
                pass

            conv_html = self._convergence_badge_html(state.get("solver_state_"))

            diagnostics = f"""<span style="margin-right: 15px;"><strong>Neurons:</strong> {n_neurons}</span>
            </div>
            <div style="margin-top: 8px;">
                <span style="margin-right: 15px;"><strong>Features:</strong> {n_features}</span>
                {conv_html}
            </div>"""

        nemos_html = f"""
        <div style="
            font-family: sans-serif;
            margin-bottom: 10px;
            padding: 10px 14px;
            border-left: 4px solid {state_color};
            background-color: #f8f9fa;
            color: #333;
            border-radius: 4px;
            display: inline-block;
            font-size: 13px;
        ">
            <div>
                <span style="font-weight: bold; margin-right: 15px;">
                    Model State: <span style="color: {state_color};">{state_text}</span>
                </span>
                {diagnostics}
        </div>
        """

        bundle["text/html"] = nemos_html + bundle.get("text/html", "")
        return bundle

    def _repr_html_(self) -> str:
        """HTML representation of the model."""
        return self._repr_mimebundle_().get("text/html", repr(self))
