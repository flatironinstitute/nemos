"""GLM-HMM for Classification."""

from ..glm.classifier_glm import ClassifierMixin
from .glm_hmm import GLMHMM
from typing import Optional, Callable, Union, Any
import jax.numpy as jnp
from ..regularizer import Regularizer
from .initialize_parameters import GLMHMM_INITIALIZATION_FN_DICT
from ..hmm.initialize_parameters import HMM_INITIALIZATION_FN_DICT
import jax
from ..observation_models import CategoricalObservations
from numpy.typing import ArrayLike, NDArray
from ..typing import DESIGN_INPUT_TYPE
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
    n_classes
        The number of classes. Must be >= 2.
    inverse_link_function
        The inverse link function. Default is ``log_softmax``.
    regularizer
        Regularization scheme used in the M-step for the per-state GLM coefficients.
        Default is ``Ridge``. Note that the model is over-parameterized: one set of
        coefficients for each class. Regularization makes the parameters identifiable.
        Setting ``UnRegularized`` will result in non-identifiable coefficients, see note below.
    regularizer_strength :
        Strength of the regularization applied to the GLM coefficients. Default is
        ``1.0``. Ignored when ``regularizer="UnRegularized"``.
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
    hmm_initialization_funcs : dict, optional
        Dictionary of initialization functions for HMM probabilities (initial and
        transition). Included for scikit-learn compatibility; prefer configuring via the
        :meth:`setup` method after construction. If ``None``, defaults from
        ``DEFAULT_INIT_FUNCTIONS`` are used.
    model_initialization_funcs : dict, optional
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
    you must call :meth:`set_classes` before using :meth:`predict`, :meth:`predict_proba`,
    :meth:`simulate`, :meth:`score`, or :meth:`compute_loss`.

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

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray:
        self._label_encoder.check_classes_is_set("score")
        y = self._label_encoder.encode(y)
        return super().score(X, y, session_starts)
