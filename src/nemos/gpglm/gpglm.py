"""Gaussian process generalized linear models (GPGLM)."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpy.typing import ArrayLike

from .. import observation_models as obs
from ..base_regressor import BaseRegressor
from ..typing import DESIGN_INPUT_TYPE
from .params import GPParams
from .samplers import Roberts

GPGLM_NOISE_TYPES = Union[
    obs.PoissonObservations,
    obs.GaussianObservations,
    obs.NegativeBinomialObservations,
    Literal["Poisson", "Gaussian", "NegativeBinomial"],
]


GPGLM_MAP_TYPES = Union[
    obs.PoissonObservations,
    Literal["FourierGP"],
]


def softplus(x):
    """Will remove when link function builder is merged."""
    return jax.nn.softplus(x)


def identity(x):
    """Will remove when link function builder is merged."""
    return x


GPGLM_LINK_TYPES = Union[
    Callable,
    Literal["Linear", "Softplus"],
]

GPGLM_LINK_REGISTRY = {
    "Linear": identity,
    "Softplus": softplus,
}


GPUserParams = Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]

GPValidator = None


class BaseGP(BaseRegressor[GPUserParams, GPParams, GPValidator]):
    """Abstract base for GPLVM-family errors-in-variables models.

    Must define `fit`, `predict`, `score`, `simulate`, `update`,
    and `compute_loss` public methods, as well as `_compute_loss`,
    `_get_model_params`, `_set_model_params`, `save_params`, and
    `_get_optimal_solver_params_config` private methods.
    `_initialize_optimizer_and_state` and `_model_specific_initialization`
    also required.

    Eventual subclasses: `GPGLM`, `EIV`, `GPLVM`, `DynamicGPLVM`.

    Parameters
    ----------
    mapping :
        Basis approximating the Gaussian process that maps the input ``X`` to features.
        Must provide an ``evaluate`` method and an ``n_basis_funcs`` attribute.
        Default is ``"FourierGP"``.
    observation_model :
        Observation model to use. The model describes the distribution of the neural activity.
        Default is the Poisson model. Alternatives are ``"Gaussian"`` and ``"NegativeBinomial"``.
    inverse_link_function :
        A function that maps the linear combination of the mapped features into a firing rate.
        Can be a callable or one of the following strings: ``"Linear"``, ``"Softplus"``.
        Default is ``"Softplus"``.
    n_neurons :
        Number of neurons to model. Used to initialize the ``mapping_params``.
    solver_name :
        Solver to use for model optimization. Defines the optimization scheme and related parameters.
        Default is ``"LBFGS"``.
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g. stepsize, tol, acceleration, etc.

    Attributes
    ----------
    mapping_params_ :
        Fitted weights of the basis functions, shape ``(n_basis_funcs, n_neurons)``.
    solver_state_ :
        State of the solver after fitting. May include details like optimization error.
    """

    def __init__(
        self,
        mapping: GPGLM_MAP_TYPES = "FourierGP",
        # TODO - make it a required basis, not jsut a string. GPGLM_MAP_TYPES = "FourierGP",
        observation_model: GPGLM_NOISE_TYPES = "Poisson",
        inverse_link_function: GPGLM_LINK_TYPES = "Softplus",
        n_neurons: int = None,
        # num_samples: int = 2 ** 10, # TODO - only really necessary for EIV
        solver_name: str = "LBFGS",
        solver_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            regularizer="UnRegularized",
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

        self.mapping = mapping
        self.observation_model = observation_model

        self.inverse_link_function = GPGLM_LINK_REGISTRY[inverse_link_function]

        self.n_neurons = n_neurons
        # self.num_samples = num_samples
        self.mapping_params_: Optional[Any] = None
        self.noise_params_: Optional[Any] = None
        self._optimizer_init_state = None
        self.solver_state_ = None
        self.weights_ = None
        self._current_fit_key = jax.random.PRNGKey(0)

    def marginal_log_likelihood_params(self, params, y, random_key):
        """Compute the log-likelihood of the neural activity, marginalized over any latent variables.

        Must be implemented by subclasses. Models without latent variables (like ``GPGLM``)
        return the conditional log-likelihood ``p(y | X)`` directly.

        Parameters
        ----------
        params :
            GPParams containing the basis function weights.
        y :
            Target neural activity. Shape ``(n_samples, n_neurons)``.
        random_key :
            jax.random.key used by subclasses that marginalize by sampling.

        Returns
        -------
        :
            The marginal log-likelihood.

        Raises
        ------
        NotImplementedError
            If called on ``BaseGP`` directly.

        """
        raise NotImplementedError(
            "marginal_log_likelihood_params must be implemented by subclass."
        )

    def single_eval(self, mapping_params, phi):
        """Predict the rate of a single neuron from the mapped features.

        Parameters
        ----------
        mapping_params :
            Basis function weights for one neuron. Shape ``(n_basis_funcs,)``.
        phi :
            Input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.

        Returns
        -------
        :
            The predicted rate of the neuron. Shape ``(n_samples,)``.

        """
        res = phi @ mapping_params
        return self.inverse_link_function(res)

    def eval_over_neurons(self, params, phi):
        """Predict the rates of all neurons from the mapped features.

        Vectorizes ``single_eval`` over neurons. Like ``_predict`` in ``GLM``, it does not
        perform any input validation, assuming that the inputs are pre-validated.

        Parameters
        ----------
        params :
            GPParams, or an array of basis function weights of shape
            ``(n_basis_funcs, n_neurons)``.
        phi :
            Input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.

        Returns
        -------
        :
            The predicted rates. Shape ``(n_samples, n_neurons)``.

        """
        if isinstance(params, GPParams):
            params = params.mapping_params
        return jax.vmap(self.single_eval, in_axes=(-1, None), out_axes=1)(params, phi)

    def _predict(self, params, y, phi):
        """Compute the log-likelihood of the neural activity given the mapped features.

        This method predicts the rates from ``phi`` and evaluates the log-likelihood of ``y``
        under the observation model, summed over samples. It is a streamlined version used
        internally within optimization routines. Unlike ``score``, it does not perform any
        input validation, assuming that the inputs are pre-validated.

        Parameters
        ----------
        params :
            GPParams containing the basis function weights.
        y :
            Target neural activity. Shape ``(n_samples, n_neurons)``.
        phi :
            Input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.

        Returns
        -------
        :
            The log-likelihood of ``y``, summed over samples.

        """
        # previously logp_y_given_x
        loc = self.eval_over_neurons(params, phi)
        # self.observation_model.log_likelihood(y, loc)
        return self.observation_model.log_likelihood(
            y, loc, aggregate_sample_scores=jnp.sum
        )

    @partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0)
    def logp_x(self, params, y, X):
        """Compute the normalized log posterior of the input over a grid of candidate values.

        For each observation in ``y``, evaluates the log-likelihood of the population
        activity at every candidate input in ``X``, sums over neurons (which are
        conditionally independent given the input), and normalizes so that the density
        sums to one over ``X``.

        Parameters
        ----------
        params :
            GPParams containing the basis function weights.
        y :
            Neural activity at a single sample. Shape ``(n_neurons,)``.
        X :
            Grid of candidate input values.

        Returns
        -------
        :
            The normalized log density over the candidate inputs.

        """
        phi = self.mapping.evaluate(X)
        y_pred = self.eval_over_neurons(params, phi)

        lps = self.observation_model.log_likelihood(
            y[:, *([jnp.newaxis] * X.shape[-1])],
            y_pred,
            aggregate_sample_scores=lambda x: x,
        )

        # Sum over neurons (conditionally independent given x).
        logp_unnrm = jnp.sum(lps, axis=1)

        # Normalize log density so that the density sums to one.
        logp = logp_unnrm - logsumexp(logp_unnrm)

        return logp

    def update(self, params, opt_state, X, y, random_key=None, **kwargs):
        """Update the model parameters and solver state.

        This method performs a single optimization step using the model's current solver.
        It updates the model's basis function weights based on the provided parameters,
        predictors (X), responses (y), and the current optimization state. This method is
        particularly useful for iterative model fitting, especially in scenarios where model
        parameters need to be updated incrementally, such as online learning or when dealing
        with very large datasets that do not fit into memory at once.

        Parameters
        ----------
        params :
            The current model parameters.
        opt_state :
            The current state of the optimizer, encapsulating information necessary for the
            optimization algorithm to continue from the current state. This includes gradients,
            step sizes, and other optimizer-specific metrics.
        X :
            The input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.
        y :
            The response variable or output data corresponding to the predictors.
            Shape ``(n_samples, n_neurons)``.
        random_key :
            jax.random.key passed to the loss.
        **kwargs :
            Additional keyword arguments to be passed to the solver's update method.

        Returns
        -------
        StepResult
            A tuple containing the updated parameters, the updated optimizer state,
            and auxiliary information.

        """
        internal_params = params
        # internal_params = self._validator.validate_params(params)
        # X, y = self._validator.validate_input(X, y)

        def loss_fn(p):
            return self._compute_loss(p, X, y, random_key)

        return self.optimizer_update(
            internal_params, opt_state, loss_fn=loss_fn, **kwargs
        )

    def predict(self, X, params=None):
        """Predict rates based on fit parameters.

        Parameters
        ----------
        X :
            Input values, mapped through the basis functions before predicting.
            Shape ``(n_samples, n_input_dims)``.
        params :
            Basis function weights of shape ``(n_basis_funcs, n_neurons)``. If ``None``,
            the fitted ``mapping_params_`` are used.

        Returns
        -------
        :
            The predicted rates with shape ``(n_samples, n_neurons)``.

        """
        # TODO - add private method
        phi = self.mapping.evaluate(X)
        if params is None:
            params = self._get_model_params().mapping_params
        return self.eval_over_neurons(params, phi)

    def score(self, X, y, random_key, score_type: str = "log-likelihood"):
        """Evaluate the goodness-of-fit of the model to the observed neural data.

        This method computes the goodness-of-fit score, currently the log-likelihood of
        the neural activity under the fitted model. A higher score indicates a better fit
        of the model to the observed data.

        Parameters
        ----------
        X :
            Input values, mapped through the basis functions before scoring.
            Shape ``(n_samples, n_input_dims)``.
        y :
            Neural activity. Shape ``(n_samples, n_neurons)``.
        random_key :
            jax.random.key for subclasses that score by sampling.
        score_type :
            Type of scoring. Currently only ``"log-likelihood"`` is supported.

        Returns
        -------
        score :
            The log-likelihood of the current model.

        Raises
        ------
        ValueError
            If ``score_type`` is not supported.

        """
        phi = self.mapping.evaluate(X)
        internal_params = self._get_model_params()
        # X, y = self._validator.validate_input(X, y)
        if score_type == "log-likelihood":
            return jnp.sum(self._predict(internal_params, y, phi))
        raise ValueError(f"Unsupported score_type: {score_type!r}")

    def save_params(self, filename: Union[str, Path]):
        """Save GPGLM model parameters to a .npz file.

        This method allows to reuse the model parameters. The saved parameters can be loaded back
        into a GPGLM instance using the ``load_model`` function.

        Parameters
        ----------
        filename :
            The name of the file where the model parameters will be saved. The file will be saved in ``.npz`` format.

        """
        # initialize saving dictionary
        fit_attrs = self._get_fit_state()
        fit_attrs.pop("solver_state_")
        self._save_params(filename, fit_attrs)

    def _get_model_params(self) -> GPParams:
        """Pack ``mapping_params_`` into a params pytree."""
        return GPParams(mapping_params=self.mapping_params_)

    def _set_model_params(self, params: GPParams):
        """Unpack and store params pytree to ``mapping_params_``."""
        self.mapping_params_ = params.mapping_params

    def _get_optimal_solver_params_config(self):
        """Return the functions for computing default step and batch size for the solver."""
        return False, None, None

    def _initialize_optimizer_and_state(self, init_params, X, y):
        """Initialize the solver by instantiating its init_state, update and, run methods.

        This method also prepares the solver's state by using the initialized model parameters and data.
        This setup is ready to be used for running the solver's optimization routines.

        Parameters
        ----------
        init_params :
            Initial parameters for the model.
        X :
            The input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.
        y :
            The response variables or outputs corresponding to the predictors.
            Shape ``(n_samples, n_neurons)``.

        Returns
        -------
        SolverState
            The initialized solver state.

        """

        def loss(params, X, y):
            return self._compute_loss(params, X, y, random_key=self._current_fit_key)

        opt_solver_kwargs = self._optimize_solver_params(X, y)
        self._solver = self._instantiate_solver(
            loss, init_params=init_params, solver_kwargs=opt_solver_kwargs
        )
        self._optimizer_init_state = self._solver.init_state
        self._optimizer_update = self._solver.update
        self._optimizer_run = self._solver.run
        return self._optimizer_init_state(init_params, X, y)

    def _compute_loss(self, params, phi, y, random_key=None, *args, **kwargs):
        """Compute the negative log posterior of the parameters given the neural activity.

        This method computes the negative of the marginal log-likelihood plus a standard
        normal log prior on the basis function weights. Unlike ``score``, it does not conduct
        parameter checks prior to evaluation. Passed directly to the solver, it serves to
        establish the optimization objective for learning the model parameters.

        Parameters
        ----------
        params :
            GPParams containing the basis function weights.
        phi :
            Input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.
        y :
            Target neural activity. Shape ``(n_samples, n_neurons)``.
        random_key :
            jax.random.key passed to the marginal log-likelihood. If ``None``, the key
            stored at fit time is used.

        Returns
        -------
        :
            The model negative log posterior. Shape (1,).

        """
        # negative log posterior params
        if random_key is None:
            random_key = self._current_fit_key

        log_lik = jnp.sum(self.marginal_log_likelihood_params(params, y, phi))
        log_prior = jax.scipy.stats.norm.logpdf(params.mapping_params).sum()
        return -(log_lik + log_prior)


class GPGLM(BaseGP):
    """Gaussian process generalized linear model (GPGLM) for neural activity data.

    This model describes the activity of a population of neurons as a function of an
    observed input variable ``X`` (for example, an animal's position). ``X`` is mapped
    through a set of basis functions approximating a Gaussian process, combined linearly
    with the ``mapping_params``, and passed through an inverse link function to obtain the
    rate of each neuron. Unlike latent variable models in the same family, ``X`` is
    observed, so no marginalization over the input is needed.

    Parameters
    ----------
    mapping :
        Basis approximating the Gaussian process that maps the input ``X`` to features.
        Must provide an ``evaluate`` method and an ``n_basis_funcs`` attribute.
        Default is ``"FourierGP"``.
    observation_model :
        Observation model to use. The model describes the distribution of the neural activity.
        Default is the Poisson model. Alternatives are ``"Gaussian"`` and ``"NegativeBinomial"``.
    inverse_link_function :
        A function that maps the linear combination of the mapped features into a firing rate.
        Can be a callable or one of the following strings: ``"Linear"``, ``"Softplus"``.
        Default is ``"Linear"``.
    n_neurons :
        Number of neurons to model. Used to initialize the ``mapping_params``.
    solver_name :
        Solver to use for model optimization. Defines the optimization scheme and related parameters.
        Default is ``"LBFGS"``.
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g. stepsize, tol, acceleration, etc.
    sampler :
        Sampler used by ``simulate`` to draw input values. Default is ``Roberts()``.

    Attributes
    ----------
    mapping_params_ :
        Fitted weights of the basis functions, shape ``(n_basis_funcs, n_neurons)``.
    solver_state_ :
        State of the solver after fitting. May include details like optimization error.
    aux_ :
        Auxiliary information returned by the solver after fitting.
    """

    def __init__(
        self,
        mapping: GPGLM_MAP_TYPES = "FourierGP",  # TODO change this
        observation_model: GPGLM_NOISE_TYPES = "Poisson",
        inverse_link_function: GPGLM_LINK_TYPES = "Linear",
        n_neurons: int = None,
        # num_samples: int = 2 ** 10,
        solver_name: str = "LBFGS",
        solver_kwargs: Optional[dict] = None,
        sampler=None,
    ):
        super().__init__(
            mapping=mapping,
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            n_neurons=n_neurons,
            # num_samples=num_samples,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

        self.sampler = sampler if sampler is not None else Roberts()

    def fit(self, X, y, init_params=None, **kwargs):
        """Fit GPGLM to neural activity.

        Fit and store the model parameters as attribute ``mapping_params_``.

        Parameters
        ----------
        X :
            Input values, array of shape ``(n_samples, n_input_dims)``. Mapped through the
            basis functions before fitting.
        y :
            Target neural activity arranged in a matrix, shape ``(n_samples, n_neurons)``.
        init_params :
            Initial parameter values. If None, the basis function weights are initialized
            by sampling from a standard normal distribution.

        Returns
        -------
        :
            The fitted model.
        """
        # phi has to be evaluated here in order to avoid concretization error,
        # but should be evaluated in marginal_ll_params
        phi = self.mapping.evaluate(jnp.asarray(X))

        if init_params is None:
            init_params = self._model_specific_initialization(phi, y)
        else:
            init_params = self._validator.validate_and_cast_params(init_params)

        self._initialize_optimizer_and_state(init_params, phi, y)
        params, state, aux = self._optimizer_run(init_params, phi, y)

        self._set_model_params(params)
        self.solver_state_ = state
        self.aux_ = aux
        return self

    def simulate(self, params, key, observations):
        """Simulate neural activity at input values drawn from the sampler.

        Parameters
        ----------
        params :
            Basis function weights, GPParams or an array of shape ``(n_basis_funcs, n_neurons)``.
        key :
            jax.random.key for seeding the simulation.
        observations :
            Number of input values to draw and simulate.

        Returns
        -------
        x :
            Simulated input values drawn from the sampler.
        y :
            Simulated observations (spike counts for a Poisson GPGLM) at each input value.
            Shape: ``(observations, n_neurons)``.
        """
        k1, k2 = jax.random.split(key, 2)
        x = self.sampler.sample(k1, observations)

        phi = self.mapping.evaluate(x)

        loc = self.eval_over_neurons(params, phi)
        y = self.observation_model.sample_generator(k2, loc)
        return x, y

    def _model_specific_initialization(self, X, y):
        """Initialize the parameters based on the structure and dimensions X and y.

        The basis function weights are initialized by sampling from a standard normal
        distribution, with shape ``(n_basis_funcs, n_neurons)``.

        Parameters
        ----------
        X :
            The input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.
        y :
            The target data array of shape ``(n_samples, n_neurons)``.

        Returns
        -------
        GPParams
            The initialized parameters.
        """
        key = getattr(self, "_current_fit_key", jax.random.PRNGKey(0))
        mapping_params = jax.random.normal(
            key, shape=(self.mapping.n_basis_funcs, self.n_neurons)
        )
        return GPParams(mapping_params=mapping_params)

    def marginal_log_likelihood_params(self, params, y, phi):
        """Compute the log-likelihood of the neural activity given the input.

        ``GPGLM`` is not a latent variable model, so no marginalization is needed and
        this is the standard conditional log-likelihood ``p(y | X)``.

        Parameters
        ----------
        params :
            GPParams containing the basis function weights.
        y :
            Target neural activity. Shape ``(n_samples, n_neurons)``.
        phi :
            Input mapped through the basis functions. Shape ``(n_samples, n_basis_funcs)``.

        Returns
        -------
        :
            The log-likelihood of ``y``, summed over samples.
        """
        return self._predict(params, y, phi)
