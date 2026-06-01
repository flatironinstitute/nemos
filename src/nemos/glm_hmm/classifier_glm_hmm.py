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
from .params import GLMHMMParams, GLMHMMUserParams


class ClassifierGLMHMM(GLMHMM, ClassifierMixin):
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
        self.n_classes = n_classes
        super().__init__(
            n_states=n_states,
            observation_model=CategoricalObservations(),
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

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[GLMHMMUserParams] = None,
        session_starts: Optional[jnp.ndarray] = None,
    ) -> "ClassifierGLMHMM":
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
