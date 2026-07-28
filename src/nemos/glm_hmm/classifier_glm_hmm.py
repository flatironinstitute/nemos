"""GLM-HMM for Classification."""

from ..glm.classifier_glm import ClassifierMixin
from .glm_hmm import GLMHMM
from ..hmm.hmm import BaseHMM
from ..basis._composition_utils import add_docstring
from typing import Optional, Callable, Union, Any, Literal, NamedTuple, Tuple
import jax.numpy as jnp
from ..regularizer import Regularizer
from .initialize_parameters import GLMHMM_INITIALIZATION_FN_DICT
from ..hmm.initialize_parameters import HMM_INITIALIZATION_FN_DICT
import jax
from ..observation_models import CategoricalObservations
from numpy.typing import ArrayLike, NDArray
from ..type_casting import support_pynapple
from ..typing import DESIGN_INPUT_TYPE, StepResult
import pynapple as nap
from .params import GLMHMMUserParams
from .validation import ClassifierGLMHMMValidator


class ClassifierGLMHMM(ClassifierMixin, GLMHMM):
    """
    Generalized Linear Model with Hidden Markov Model (GLM-HMM) for multi-class classification.

    This model combines a Generalized Linear Model (GLM) with a Hidden Markov Model (HMM) to capture
    state-dependent relationships between predictors and neural or behavioral responses. The GLM-HMM
    is suitable for modeling time series data where the relationship between inputs and outputs
    varies according to an underlying latent state that evolves over time following Markovian dynamics.

    The model assumes that at each time step, the system is in one of ``n_states`` discrete hidden states.
    Each state has its own GLM parameters (coefficients and intercept), and transitions between states
    are governed by a transition probability matrix. The model is fitted using the Expectation-Maximization
    (EM) algorithm.

    This version of the GLM-HMM is specifically designed for classification tasks, using a softmax
    (multinomial logistic) model to link features to a discrete set of class labels. It uses an
    over-parameterized representation with one set of coefficients per class, resulting in
    coefficient shape ``(n_features, n_classes, n_states)`` and intercept shape ``(n_classes, n_states)``.

    Below is a table listing the default and available solvers for each regularizer.

    +---------------+------------------+-------------------------------------------------------------+
    | Regularizer   | Default Solver   | Available Solvers                                           |
    +===============+==================+=============================================================+
    | UnRegularized | LBFGS            | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Ridge         | LBFGS            | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Lasso         | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+
    | ElasticNet    | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+
    | GroupLasso    | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+

    Parameters
    ----------
    n_states :
        The number of hidden states in the HMM. Must be a positive integer.
    n_classes :
        The number of classes. Must be >= 2.
    inverse_link_function :
        The inverse link function. Default is ``log_softmax``.
    regularizer :
        Regularization scheme used in the M-step for the per-state GLM coefficients.
        Default is ``Ridge``. Note that the model is over-parameterized: one set of
        coefficients for each class. Regularization makes the parameters identifiable.
        Setting ``UnRegularized`` will result in non-identifiable coefficients, see note below.
    regularizer_strength :
        Strength of the regularization applied to the GLM coefficients. Default is
        ``1.0``. Ignored when ``regularizer="UnRegularized"``.
    dirichlet_initial_proba :
        Alpha parameters for the Dirichlet prior over the initial state probabilities.
        Shape ``(n_states,)``. If None, a flat (uninformative) prior is assumed.
    dirichlet_transition_proba :
        Alpha parameters for the Dirichlet prior over the transition probabilities.
        Shape ``(n_states, n_states)``. If None, a flat (uninformative) prior is assumed.
    solver_name :
        Solver used for the GLM M-step. The solver must be valid for the chosen
        regularizer (see table above). Default is ``None``, in which case the
        regularizer's default solver is selected (``"LBFGS"`` for Ridge /
        UnRegularized, ``"ProximalGradient"`` for Lasso / ElasticNet /
        GroupLasso).
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g., stepsize, tol, acceleration, etc.
    maxiter :
        Maximum number of EM iterations. Default is 1000.
    tol :
        Convergence tolerance for the EM algorithm. The algorithm stops when the absolute change
        in log-likelihood between consecutive iterations falls below this threshold. Default is 1e-8.
    seed :
        JAX PRNG key for random number generation during initialization. Default is
        ``jax.random.PRNGKey(123)``.
    hmm_initialization_funcs :
        Dictionary of initialization functions for HMM probabilities (initial and
        transition). Included for scikit-learn compatibility; prefer configuring via the
        :meth:`setup` method after construction. If ``None``, defaults from
        ``DEFAULT_INIT_FUNCTIONS`` are used.
    model_initialization_funcs :
        Dictionary of initialization functions for the GLM-specific parameters
        (coefficients, intercept, and scale). Included for scikit-learn compatibility;
        prefer configuring via the :meth:`setup` method after construction. If ``None``,
        defaults from ``DEFAULT_INIT_FUNCTIONS_GLMHMM`` are used.

    Attributes
    ----------
    transition_prob_ :
        Transition probability matrix of shape ``(n_states, n_states)``. Entry ``[i, j]`` represents
        the probability of transitioning from state ``i`` to state ``j``.
    initial_prob_ :
        Initial state probability vector of shape ``(n_states,)``. Entry ``[i]`` represents
        the probability of starting in state ``i``.
    coef_ :
        GLM coefficients for each state, shape ``(n_features, n_classes, n_states)``.
    intercept_ :
        GLM intercepts (bias terms) for each state, shape ``(n_classes, n_states)``.
    solver_state_ :
        State of the solver after fitting. May include details like optimization error.
    scale_ :
        Scale parameter for the observation model, shape ``(n_states,)``.
    dof_resid_ :
        Degrees of freedom for the residuals.

    Notes
    -----
    To bypass the initialization functions entirely and provide parameter arrays
    directly, pass them to the ``fit()`` method::

        model.fit(X, y, init_params=my_params)

    **Identifiability**

    This model uses an over-parameterized (symmetric) representation where each class
    has its own set of coefficients. Since probabilities from softmax are invariant to
    adding a constant to all linear predictors, the parameters are not uniquely
    identifiable without regularization. For example, if ``(coef, intercept)`` is a
    solution, so is ``(coef + c, intercept + c)`` for any constant ``c``.

    Using regularization (default is ``Ridge``) resolves this ambiguity by penalizing
    the parameter magnitudes, effectively centering the solution. If you use
    ``UnRegularized``, the optimization may converge to different equivalent solutions
    depending on initialization, though predictions will be identical.

    **Class Labels**

    The target array ``y`` can contain any hashable class labels that can be stored
    in a NumPy array, including integers, strings, or other hashable types. The model
    internally maps these labels to indices ``[0, n_classes - 1]`` for computation
    and maps them back when returning predictions.

    **Performance Considerations**

    For optimal performance, use integer labels ``[0, 1, ..., n_classes - 1]``. When
    labels follow this convention, the model skips the encoding/decoding steps entirely.
    Using other label formats (e.g., ``["cat", "dog"]`` or ``[5, 10, 15]``) incurs a
    small overhead for label translation.

    **Setting Class Labels**

    The :meth:`fit` and :meth:`initialize_optimizer_and_state` methods automatically infer
    class labels from the provided ``y``. If you set ``coef_`` and ``intercept_`` manually,
    you must call :meth:`set_classes` before using :meth:`decode_state`, :meth:`smooth_proba`,
    :meth:`filter_proba`, :meth: `update`, :meth:`simulate`, :meth:`score`, or :meth:`compute_loss`.

    Raises
    ------
    TypeError
        If ``n_states`` is not a positive integer.
    TypeError
        If provided ``regularizer`` is not valid.
    TypeError
        If ``seed`` is not a valid JAX PRNG key.
    KeyError
        If ``hmm_initialization_funcs`` or ``model_initialization_funcs`` contains keys
        that are not valid for their respective default dictionary.
    ValueError
        If any ``*_kwargs`` entry in either initialization-funcs dictionary contains
        keyword arguments that don't match the signature of the corresponding
        initialization function.
    ValueError
        If ``maxiter`` is not a positive integer.
    ValueError
        If ``tol`` is not a positive float.

    See Also
    --------
    GLMHMM : Generalized Linear Model with Hidden Markov Model (GLM-HMM) for other observation models.

    Examples
    --------
    **Fit a ClassifierGLMHMM**

    The number of hidden states is the only required argument; the number of classes can be inferred
    from the observations. The shape of ``coef_`` depends on the number of features, classes, and states, 
    and the HMM transition matrix and initial distribution are exposed as fitted attributes.

    >>> import jax
    >>> import numpy as np
    >>> import nemos as nmo
    >>> np.random.seed(123)
    >>> X = np.random.normal(size=(200, 5))
    >>> # Simulate binary labels for a 2-class problem
    >>> y = np.random.binomial(n=1, p=0.5, size=200)
    >>> model = nmo.glm_hmm.ClassifierGLMHMM(n_states=3).fit(X, y)
    >>> model.coef_.shape
    (5, 2, 3)
    >>> model.transition_prob_.shape
    (3, 3)
    >>> model.initial_prob_.shape
    (3,)

    **Multi-class Classification**

    Classify into more than two classes:

    >>> np.random.seed(123)
    >>> key = jax.random.PRNGKey(123)
    >>> rate = np.random.normal(size=(200, 4))
    >>> y = jax.random.categorical(key, rate)
    >>> model = nmo.glm_hmm.ClassifierGLMHMM(n_states=3, n_classes=4).fit(X, y)
    >>> model.coef_.shape
    (5, 4, 3)

    **Fit Across Multiple Sessions**

    Mark session boundaries with ``session_starts`` so the HMM resets at each
    new session start instead of treating the data as a single chain. Pass
    either a boolean mask of shape ``(n_time_bins,)`` with ``True`` at each
    session start, or an integer array of session-start indices — the two
    are equivalent:

    >>> is_new_mask = np.zeros(200, dtype=bool)
    >>> is_new_mask[0] = True
    >>> is_new_mask[100] = True
    >>> model = nmo.glm_hmm.ClassifierGLMHMM(n_states=2).fit(X, y, session_starts=is_new_mask)
    >>> # Equivalent: pass the starts as integer indices.
    >>> model = nmo.glm_hmm.ClassifierGLMHMM(n_states=2).fit(X, y, session_starts=np.array([0, 100]))

    **Decode Hidden States**

    Recover the most-likely state sequence (Viterbi-style) or the smoothed
    posterior probabilities from the forward-backward pass:

    >>> states = model.decode_state(X, y, session_starts=is_new_mask)
    >>> states.shape
    (200, 2)
    >>> post = model.smooth_proba(X, y, session_starts=is_new_mask)
    >>> post.shape
    (200, 2)
    """

    _validator_class = ClassifierGLMHMMValidator

    def __init__(
        self,
        n_states: int,
        n_classes: Optional[int] = 2,
        inverse_link_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        regularizer: Union[str, Regularizer] = "Ridge",
        regularizer_strength: Any = 1.0,
        dirichlet_initial_proba: Union[jnp.ndarray, None] = None,  # (n_state, )
        dirichlet_transition_proba: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state)
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
        hmm_initialization_funcs: Optional[HMM_INITIALIZATION_FN_DICT] = None,
        model_initialization_funcs: Optional[GLMHMM_INITIALIZATION_FN_DICT] = None,
    ):
        # set _n_states before n_classes so validator can access it
        self._set_n_states(n_states)
        self.n_classes = n_classes
        super().__init__(
            n_states=n_states,
            observation_model=CategoricalObservations(class_axis=-2),
            inverse_link_function=inverse_link_function,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            dirichlet_initial_proba=dirichlet_initial_proba,
            dirichlet_transition_proba=dirichlet_transition_proba,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            hmm_initialization_funcs=hmm_initialization_funcs,
            model_initialization_funcs=model_initialization_funcs,
        )

    def _get_validator_extra_params(self) -> dict:
        """Get validator extra parameters."""
        return {"n_classes": self._label_encoder.n_classes, "n_states": self._n_states}

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[GLMHMMUserParams] = None,
        session_starts: Optional[jnp.ndarray] = None,
    ) -> "ClassifierGLMHMM":
        """Fit the Classifier GLM-HMM via Expectation-Maximization.

        Runs the EM algorithm until the absolute change in log-likelihood between
        consecutive iterations falls below ``tol`` or ``maxiter`` is reached.
        Fitted parameters are exposed on the instance as ``coef_``, ``intercept_``,
        ``scale_``, ``initial_prob_``, ``transition_prob_``, plus
        ``solver_state_`` (EM trace) and ``dof_resid_``.

        How parameters are initialized:

        - If ``init_params`` is ``None`` (typical), the per-state GLM parameters
          and HMM probabilities are produced by the initializers configured via
          :meth:`setup` (or the package defaults when :meth:`setup` was never
          called).
        - If ``init_params`` is provided, it bypasses the initializers entirely.
          It must be a 5-tuple ``(coef, intercept, scale, initial_prob,
          transition_prob)`` whose shapes are consistent with ``X``, ``y``, and
          ``n_states``.

        Parameters
        ----------
        X :
            Predictors, shape ``(n_time_bins, n_features)``. A pytree of arrays
            sharing leading dimension is also accepted; the fitted ``coef_``
            mirrors the pytree structure (with a trailing state axis). A pynapple
            ``TsdFrame`` is accepted.
        y :
            Observations, shape ``(n_time_bins,)`` for single neuron or
            ``(n_time_bins, n_neurons)`` for population models. A pynapple
            ``Tsd``/``TsdFrame`` is accepted.
        init_params :
            Optional explicit initial parameters as a 5-tuple
            ``(coef, intercept, scale, initial_prob, transition_prob)``. When
            ``None`` (default), the initializers configured by :meth:`setup`
            (or the defaults) are used.
        session_starts :
            Optional session boundaries for the HMM. Accepts:

            - a boolean array of shape ``(n_time_bins,)`` with ``True`` at each
              session start,
            - an integer array of session-start indices,
            - a pynapple ``IntervalSet`` (requires ``X`` or ``y`` to be a
              pynapple object to supply timestamps).

            If ``X`` or ``y`` is a pynapple object and ``session_starts`` is
            ``None``, the (unique, enforced) ``time_support`` of the pynapple
            input determines the session starts. With no pynapple input and
            ``session_starts=None``, the whole input is treated as a single
            session.

        Returns
        -------
        self :
            The fitted estimator.

        Raises
        ------
        ValueError
            If inputs fail dimensionality, shape, or consistency checks (e.g.
            ``coef`` features do not match ``X.shape[1]``, or NaNs appear
            mid-epoch).
        TypeError
            If ``init_params`` is not a 5-tuple or has incompatible leaf types.

        Warns
        -----
        RuntimeWarning
            Emitted when EM runs out of iterations without satisfying the ``tol``
            criterion (``solver_state_.iterations == maxiter``). Consider
            enabling float64, raising ``maxiter``, or loosening ``tol``.

        Examples
        --------
        Basic fit with default Bernoulli observations:

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(0)
        >>> X = np.random.normal(size=(200, 4))
        >>> y = np.random.binomial(n=1, p=0.5, size=200)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y)
        >>> model.coef_.shape, model.transition_prob_.shape
        ((4, 2), (2, 2))

        Multiple sessions via explicit ``session_starts``:

        >>> session_starts = np.array([0, 100])
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y, session_starts=session_starts)

        See Also
        --------
        setup : Configure the initializers used when ``init_params is None``.
        update : Run a single EM iteration (advanced, manual loop).
        """
        self.set_classes(y)
        y = self._label_encoder.encode(y)
        return super().fit(X, y, init_params, session_starts)

    @add_docstring("score", BaseHMM)
    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray:
        self._label_encoder.check_classes_is_set("score")
        y = self._label_encoder.encode(y)
        return super().score(X, y, session_starts)

    def decode_state(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        session_starts: Optional[ArrayLike] = None,
        state_format: Literal["one-hot", "index"] = "one-hot",
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the most likely hidden state sequence (Viterbi decoding).

        Finds the single most likely sequence of hidden states that best explains
        the observed data. Uses the Viterbi (max-sum) algorithm to compute the
        state sequence that maximizes the joint probability of states and observations.

        Unlike :meth:`smooth_proba` and :meth:`filter_proba`, which return a
        probability distribution over states at each time bin, this method makes
        a hard assignment to the single globally optimal state path.

        The decoded states answer: "What is the most likely sequence of states
        that generated the observed data?"

        Parameters
        ----------
        X :
            Predictors, shape ``(n_time_bins, n_features)``. A pytree of 2-D
            arrays sharing the leading time axis is also accepted.
        y :
            Observations, shape ``(n_time_bins,)`` for a single neuron or
            ``(n_time_bins, n_neurons)`` for a population model. A pynapple
            ``Tsd``/``TsdFrame`` is accepted; session boundaries are then
            inferred from ``time_support``.
        session_starts :
            Optional session boundaries. Accepts:

            - a boolean array of shape ``(n_time_bins,)`` with ``True`` at each
              session start,
            - an integer array of session-start indices,
            - a pynapple ``IntervalSet`` (requires ``X`` or ``y`` to be a
              pynapple object to supply timestamps).

            If ``None``, the entire input is treated as a single session.
        state_format :
            Format of the returned state sequence:

            - ``"one-hot"`` (default): binary array of shape
              ``(n_time_bins, n_states)`` with a single 1 per row.
            - ``"index"``: integer array of shape ``(n_time_bins,)`` with
              values in ``[0, n_states - 1]``.

        Returns
        -------
        decoded_states :
            Most likely state sequence. Shape and dtype depend on
            ``state_format`` (see above). Returns a pynapple ``TsdFrame``
            (columns ``"state_0"``, ``"state_1"``, …) for ``"one-hot"`` format
            or a pynapple ``Tsd`` for ``"index"`` format when the inputs are
            pynapple objects; otherwise returns a JAX array.

        Raises
        ------
        ValueError
            If the model has not been fitted (call :meth:`fit` first).
        ValueError
            If ``state_format`` is not ``"one-hot"`` or ``"index"``.
        ValueError
            If ``X`` or ``y`` contain NaN values in the interior of an epoch
            (boundary NaNs are allowed and removed before inference).
        ValueError
            If ``X`` and ``y`` have inconsistent shapes or feature counts.

        See Also
        --------
        smooth_proba :
            Compute smoothing posteriors (soft, probabilistic state assignments).
        filter_proba :
            Compute filtering posteriors (causal, conditioned on past observations).

        Notes
        -----
        Viterbi decoding finds the globally optimal state *sequence*, which can
        differ from the sequence of states that are individually most probable
        at each time bin (as returned by :meth:`smooth_proba`). For uncertainty
        estimates use :meth:`smooth_proba` instead. Session boundaries reset the
        Viterbi recursion so that no path crosses session borders.

        Examples
        --------
        Decode the most likely state sequence as integer indices:

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.poisson(2, size=100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson").fit(X, y)
        >>> states = model.decode_state(X, y, state_format="index")
        >>> states.shape
        (100,)

        One-hot output (default):

        >>> states_onehot = model.decode_state(X, y)
        >>> states_onehot.shape
        (100, 3)
        >>> bool(np.all(states_onehot.sum(axis=1) == 1))
        True
        """
        self._label_encoder.check_classes_is_set("decode_state")
        y = self._label_encoder.encode(y)
        return super().decode_state(X, y, session_starts, state_format)

    def smooth_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute smoothing posterior probabilities over hidden states.

        Computes the probability of being in each hidden state at each time bin,
        conditioned on the entire observed sequence. Uses the forward-backward
        algorithm to incorporate information from both past and future observations,
        providing optimal state estimates given all available data.

        The smoothing posteriors answer: "Given all observations, what is the
        probability that the system was in state ``k`` at time ``t``?"

        Parameters
        ----------
        X :
            Predictors, shape ``(n_time_bins, n_features)``. A pytree of 2-D
            arrays sharing the leading time axis is also accepted.
        y :
            Observations, shape ``(n_time_bins,)`` for a single neuron or
            ``(n_time_bins, n_neurons)`` for a population model. A pynapple
            ``Tsd``/``TsdFrame`` is accepted; session boundaries are then
            inferred from ``time_support``.
        session_starts :
            Optional session boundaries. Accepts:

            - a boolean array of shape ``(n_time_bins,)`` with ``True`` at each
              session start,
            - an integer array of session-start indices,
            - a pynapple ``IntervalSet`` (requires ``X`` or ``y`` to be a
              pynapple object to supply timestamps).

            If ``None``, the entire input is treated as a single session.

        Returns
        -------
        posteriors :
            Smoothing posterior probabilities, shape ``(n_time_bins, n_states)``.
            Each row sums to 1. Returns a pynapple ``TsdFrame`` (with columns
            named ``"state_0"``, ``"state_1"``, …) when the inputs are pynapple
            objects; otherwise returns a JAX array.

        Raises
        ------
        ValueError
            If the model has not been fitted (call :meth:`fit` first).
        ValueError
            If ``X`` or ``y`` contain NaN values in the interior of an epoch
            (boundary NaNs are allowed and removed before inference).
        ValueError
            If ``X`` and ``y`` have inconsistent shapes or feature counts.

        See Also
        --------
        filter_proba :
            Compute filtering posteriors (conditioned on past observations only).
        decode_state :
            Compute the most likely state sequence via Viterbi decoding.

        Notes
        -----
        Smoothing uses all data (non-causal) and gives better state estimates than
        filtering. For online or real-time applications use :meth:`filter_proba`
        instead. Session boundaries reset the HMM chain so that no information
        crosses session borders.

        Examples
        --------
        Fit a GLM-HMM and compute smoothing posteriors:

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.poisson(2, size=100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson").fit(X, y)
        >>> posteriors = model.smooth_proba(X, y)
        >>> posteriors.shape
        (100, 3)
        >>> bool(np.allclose(posteriors.sum(axis=1), 1.0))
        True

        With pynapple inputs the result is returned as a ``TsdFrame``:

        >>> import pynapple as nap
        >>> t = np.arange(100) * 0.01
        >>> X_tsd = nap.TsdFrame(t=t, d=X)
        >>> y_tsd = nap.Tsd(t=t, d=y.astype(float))
        >>> type(model.smooth_proba(X_tsd, y_tsd)).__name__
        'TsdFrame'
        """
        self._label_encoder.check_classes_is_set("smooth_proba")
        y = self._label_encoder.encode(y)
        return super().smooth_proba(X, y, session_starts=session_starts)

    def filter_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute filtering posterior probabilities over hidden states.

        Computes the probability of being in each hidden state at each time bin,
        conditioned only on observations up to that time bin. Uses the forward
        pass of the forward-backward algorithm, providing causal (online) state
        estimates that rely solely on past and current observations.

        The filtering posteriors answer: "Given observations up to time ``t``,
        what is the probability that the system is in state ``k`` at time ``t``?"

        Parameters
        ----------
        X :
            Predictors, shape ``(n_time_bins, n_features)``. A pytree of 2-D
            arrays sharing the leading time axis is also accepted.
        y :
            Observations, shape ``(n_time_bins,)`` for a single neuron or
            ``(n_time_bins, n_neurons)`` for a population model. A pynapple
            ``Tsd``/``TsdFrame`` is accepted; session boundaries are then
            inferred from ``time_support``.
        session_starts :
            Optional session boundaries. Accepts:

            - a boolean array of shape ``(n_time_bins,)`` with ``True`` at each
              session start,
            - an integer array of session-start indices,
            - a pynapple ``IntervalSet`` (requires ``X`` or ``y`` to be a
              pynapple object to supply timestamps).

            If ``None``, the entire input is treated as a single session.

        Returns
        -------
        posteriors :
            Filtering posterior probabilities, shape ``(n_time_bins, n_states)``.
            Each row sums to 1. Returns a pynapple ``TsdFrame`` (with columns
            named ``"state_0"``, ``"state_1"``, …) when the inputs are pynapple
            objects; otherwise returns a JAX array.

        Raises
        ------
        ValueError
            If the model has not been fitted (call :meth:`fit` first).
        ValueError
            If ``X`` or ``y`` contain NaN values in the interior of an epoch
            (boundary NaNs are allowed and removed before inference).
        ValueError
            If ``X`` and ``y`` have inconsistent shapes or feature counts.

        See Also
        --------
        smooth_proba :
            Compute smoothing posteriors (conditioned on all observations).
        decode_state :
            Compute the most likely state sequence via Viterbi decoding.

        Notes
        -----
        Filtering is causal: each posterior at time ``t`` uses only observations
        up to ``t``, making it suitable for online or real-time applications.
        For retrospective analysis where all data are available, :meth:`smooth_proba`
        gives better state estimates. Session boundaries reset the HMM chain so
        that no information crosses session borders.

        Examples
        --------
        Fit a GLM-HMM and compute filtering posteriors (causal/online):

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.poisson(2, size=100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson").fit(X, y)
        >>> filt = model.filter_proba(X, y)
        >>> filt.shape
        (100, 3)
        >>> bool(np.allclose(filt.sum(axis=1), 1.0))
        True

        With pynapple inputs the result is returned as a ``TsdFrame``:

        >>> import pynapple as nap
        >>> t = np.arange(100) * 0.01
        >>> X_tsd = nap.TsdFrame(t=t, d=X)
        >>> y_tsd = nap.Tsd(t=t, d=y.astype(float))
        >>> type(model.filter_proba(X_tsd, y_tsd)).__name__
        'TsdFrame'
        """
        self._label_encoder.check_classes_is_set("filter_proba")
        y = self._label_encoder.encode(y)
        return super().filter_proba(X, y, session_starts=session_starts)

    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
        state_format: Literal["one-hot", "index"] = "index",
        session_starts: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate neural activity and hidden states from the model.

        Simulates a trajectory through the hidden state space according to the
        HMM dynamics, then generates observations from the GLM emission model
        conditioned on each state.

        Parameters
        ----------
        random_key :
            JAX random key for reproducible simulation.
        feedforward_input :
            Design matrix of shape ``(n_time_bins, n_features)``. If a pynapple
            Tsd/TsdFrame is provided, session boundaries are detected from
            ``time_support`` and the hidden state chain is reset at each session start.
        state_format :
            Format for the returned states:

            - ``"index"``: Integer array of shape ``(n_time_bins,)`` with state indices.
            - ``"one-hot"``: Binary array of shape ``(n_time_bins, n_states)``.
        session_starts :
            Optional session boundaries. Accepts:

            - a boolean array of shape ``(n_time_bins,)`` with ``True`` at each
              session start,
            - an integer array of session-start indices,
            - a pynapple ``IntervalSet`` (requires ``feedforward_input`` to be a
              pynapple object to supply timestamps).

            If ``feedforward_input`` is a pynapple object and ``session_starts``
            is ``None``, the ``time_support`` determines the session starts. With
            no pynapple input and ``session_starts=None``, the whole input is
            treated as a single session.

        Returns
        -------
        simulated_activity :
            Simulated observations from the emission model. Shape ``(n_time_bins,)``
            for single neuron or ``(n_time_bins, n_neurons)`` for population models.
        firing_rates :
            Predicted firing rates conditioned on the simulated states.
            Shape ``(n_time_bins,)`` or ``(n_time_bins, n_neurons)``.
        simulated_states :
            Simulated hidden state trajectory. Shape depends on ``state_format``.

        Raises
        ------
        ValueError
            If the model has not been fit.

        Examples
        --------
        >>> import jax
        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 3)
        >>> y = np.random.binomial(1, 0.5, 100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2, observation_model="Bernoulli")
        >>> model = model.fit(X, y)
        >>> key = jax.random.key(0)
        >>> X_new = np.random.randn(50, 3)
        >>> activity, rates, states = model.simulate(key, X_new)
        >>> activity.shape
        (50,)
        >>> states.shape
        (50,)

        See Also
        --------
        decode_state : Infer most likely state sequence from observations.
        smooth_proba : Compute posterior state probabilities.
        """
        self._label_encoder.check_classes_is_set("simulate")
        y, y_proba, simulated_states = super().simulate(
            random_key, feedforward_input, state_format, session_starts
        )
        argmax = support_pynapple(conv_type="jax")(lambda x: jnp.argmax(x, axis=-2))
        y = self._label_encoder.decode(argmax(y))
        return y, y_proba, simulated_states

    def update(
        self,
        params: GLMHMMUserParams,
        opt_state: NamedTuple,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        session_starts: Optional[jnp.ndarray] = None,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """Run a single EM iteration on the GLM-HMM.

        Performs one E-step / M-step pair starting from the supplied parameters and
        EM state, updates the model's fitted attributes (``coef_``, ``intercept_``,
        ``scale_``, ``initial_prob_``, ``transition_prob_``, ``solver_state_``,
        ``dof_resid_``) in place, and returns the updated parameter tuple and EM
        state. Intended for callers that need fine-grained control over EM
        iteration (e.g. checkpointing, custom convergence criteria) instead of the
        bundled :meth:`fit` loop.

        :meth:`initialize_optimizer_and_state` must be called first so that the EM
        step function and initial ``opt_state`` are available.

        Parameters
        ----------
        params :
            Current model parameters as a 5-tuple
            ``(coef, intercept, scale, initial_prob, transition_prob)`` matching
            the structure produced by :meth:`initialize_params`.
        opt_state :
            EM state returned by :meth:`initialize_optimizer_and_state` or by the
            previous call to :meth:`update`.
        X :
            Predictors, shape ``(n_time_bins, n_features)`` (or a pytree of arrays
            of the same shape).
        y :
            Observations, shape ``(n_time_bins,)`` or ``(n_time_bins, n_neurons)``.
        session_starts :
            Optional session boundaries. Accepts:

            - a boolean array of shape ``(n_time_bins,)`` with ``True`` at each
              session start,
            - an integer array of session-start indices,
            - a pynapple ``IntervalSet`` (requires ``X`` or ``y`` to be a
              pynapple object to supply timestamps).

            If ``None``, the entire input is treated as a single session.
        n_samples :
            Total sample count to use when estimating the residual degrees of
            freedom. Defaults to ``X.shape[0]``.

        Returns
        -------
        params :
            Updated user-facing parameter tuple.
        state :
            Updated EM state.

        Raises
        ------
        ValueError
            If inputs fail shape/consistency validation.

        Examples
        --------
        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(0)
        >>> X = np.random.normal(size=(80, 3))
        >>> y = np.random.binomial(n=1, p=0.5, size=80)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2)
        >>> init_params = model.initialize_params(X, y)
        >>> opt_state = model.initialize_optimizer_and_state(init_params, X, y)
        >>> new_params, new_state = model.update(init_params, opt_state, X, y)
        """
        self._label_encoder.check_classes_is_set("update")
        y = self._label_encoder.encode(y)
        return super().update(
            params, opt_state, X, y, session_starts=session_starts, n_samples=n_samples
        )
