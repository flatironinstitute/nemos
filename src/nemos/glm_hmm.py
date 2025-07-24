from typing import Tuple
from .typing import Pytree
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
Array = NDArray | jax.numpy.ndarray
from scipy.special import logsumexp
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from nemos.observation_models import BernoulliObservations
import jax
import jax.numpy as jnp
from functools import partial
Array = NDArray | jax.numpy.ndarray
from nemos.glm_hmm_utils import forward_pass, backward_pass
jax.config.update("jax_enable_x64", True) 
from scipy.optimize import minimize


class GLM_HMM():
    # Currently assuming that it will always be a logistic link as its the case for Bernoulli
    def __init__(
        self,
    ):
        self.observation_model = BernoulliObservations()

    def run_baum_welch(
            self,
            X: Array, 
            y: Array, 
            initial_prob: Array, 
            transition_prob: Array,
            latent_weights: Array,
            new_sess: Array | None = None
    ):
        """"
        Baum-Welch algorithm to compute the forward-backward algorithm and return the marginal posterior distribution.

        According to Bishop's "Pattern Recognition and Machine Learning".

        Parameters
        ----------
        X : 
            (n_time_bins x n_features) design matrix
        y : 
            (n_time_bins,) observations
        initial_prob : .pi
            (n_states x 1) initial latent state probability
        latent_weights : .w
            (n_features x n_states) latent state GLM weights
        transition_prob : .A
            (n_states x n_states) latent state transition matrix
        new_sess :
            logical array with 1s denoting the start of a new session. If unspecified or empty, treats the full set of trials as a single session.

        Returns
        -------   
        gammas :
            (n_states x n_time_bins) marginal posterior distribution
        xis :
            (n_states x n_states x n_time_bins) joint posterior distribution
        ll :
            log-likelihood of the fit
        """
        # Initialize variables
        n_time_bins, n_features = X.shape  # n_time_bins and n_features from dimensions of X
        n_states = latent_weights.shape[1]  # number of latent states from dimensions of w

        # Revise if the data is one single session or multiple sessions. If new_sess is not provided, assume one session
        if new_sess is None:
            new_sess = np.zeros_like(y, dtype=bool)
        new_sess[0] = True

        # Firing rate
        tmpy = self.observation_model.inverse_link_function(X @ latent_weights)
        print("a", y[:,jnp.newaxis].shape)
        print("b", tmpy.shape)
        
        # Data likelihood p(y|z) from emissions model using NeMoS
        py_z = jnp.exp(
            self.observation_model.log_likelihood(
                y[:,jnp.newaxis],
                tmpy,
               aggregate_sample_scores = lambda x: x
            )
        )   
        print("c", py_z.shape)
        py_z = py_z.T
        ###### Forward recursion to compute alphas ######
        # Initialize variables
        alphas = np.full((n_states, n_time_bins), np.nan) # forward pass alphas
        c = np.full(n_time_bins, np.nan) # variable to store marginal likelihood

        for t in range(n_time_bins):
            if new_sess[t]:
                alphas[:, t] = initial_prob * py_z[:, t] # Initial alpha. Equation 13.37. Reinitialize for new sessions
            else:
                alphas[:, t] = py_z[:, t] * (transition_prob.T @ alphas[:, t - 1]) # Equation 13.36

            c[t] = np.sum(alphas[:, t]) # Store marginal likelihood
            if c[t] == 0: # This should not happen, but if it does, raise an error if weights are out of control
                raise ValueError(f"Zero marginal likelihood at time {t} - Weights may be out of control")
            alphas[:, t] /= c[t] # Normalize (Equation 13.59)

        ll = np.sum(np.log(c)) # Store log-likelihood
        ll_norm = np.exp(ll / n_time_bins)

        ###### Backward recursion to compute betas ######
        # Initialize variables
        betas = np.full((n_states, n_time_bins), np.nan) # backward pass betas
        betas[:, -1] = np.ones(n_states) # initial beta (Equation 13.39)

        # Solve for remaining betas
        for t in range(n_time_bins - 2, -1, -1):
            if new_sess[t + 1]:
                betas[:, t] = np.ones(n_states) # Reinitialize backward pass if end of session
            else:
                betas[:, t] = transition_prob @ (betas[:, t + 1] * py_z[:, t + 1]) # Equation 13.38
                betas[:, t] /= c[t + 1] # Normalize (Equation 13.62)

        ###### Compute posterior distributions ######
        gammas = alphas * betas # Gamma - Equations 13.32, 13.64

        # Trials to compute xi
        # Exclude the first trial of every session
        # Transition matrix
        trials_xi = np.arange(n_time_bins)
        trials_xi = trials_xi[~new_sess]

        # Equations 13.43 and 13.65
        # Xi summed across time steps
        xi_numer = (alphas[:, trials_xi - 1] / c[trials_xi]) @ (py_z[:, trials_xi] * betas[:, trials_xi]).T
        xis = xi_numer * transition_prob
        #print(xi_numer * transition_prob)
        return gammas, xis, ll, ll_norm, alphas, betas
                                                                                 
    def run_baum_welch_jax(
        self,
        X: Array,
        y: Array,
        initial_prob: Array,
        transition_prob: Array,
        projection_weights: Array,
        new_sess: Array | None = None,
    ):
        """
        Baum-Welch algorithm to compute the forward-backward algorithm and return the marginal posterior distribution.

        According to Bishop's "Pattern Recognition and Machine Learning".

        Parameters
        ----------
        X : 
            (n_time_bins x n_features) design matrix

        y : 
            (n_time_bins,) observations

        initial_prob : .pi
            (n_states x 1) initial latent state probability

        transition_prob : .A
            (n_states x n_states) latent state transition matrix

        projection_weights : .w
            (n_features x n_states) latent state GLM weights

        new_sess :
            logical array with 1s denoting the start of a new session. If unspecified or empty, treats the full set of trials as a single session.

        Returns
        -------
        gammas :
            (n_states x n_time_bins) marginal posterior distribution

        xis :
            (n_states x n_states x n_time_bins) joint posterior distribution

        ll :
            log-likelihood of the fit
        """
        # Initialize variables
        n_time_bins = X.shape[0]  # n_time_bins and n_features from dimensions of X

        # Revise if the data is one single session or multiple sessions. If new_sess is not provided, assume one session
        if new_sess is None:
            new_sess = jnp.zeros_like(y, dtype=bool)
        new_sess[0] = True

        # Convert new_sess to jax array
        new_sess = jnp.asarray(new_sess)
        initial_prob = jnp.asarray(initial_prob)

        # Predicted y
        tmpy = self.observation_model.inverse_link_function(X @ projection_weights)

        # Compute likelihood given the fixed weights
        # Data likelihood p(y|z) from emissions model
        py_z = jnp.exp(
            self.observation_model.log_likelihood(
                y[:, jnp.newaxis],
                tmpy,
               aggregate_sample_scores = lambda x: x
            )
        )   # TODO Will this break with other observation models?
        # Compute forward pass
        with jax.disable_jit(False):
            alphas_scan, c_scan = forward_pass(
                initial_prob, transition_prob, py_z, new_sess
            )  # these are equivalent to the forward pass with python loop
        
        #t0 = perf_counter() # Counter for benchmarking
        # Compute backward pass
        with jax.disable_jit(False):
            betas_scan = backward_pass(transition_prob, py_z, c_scan, new_sess)
        #print("\nscan", perf_counter() - t0) # Print duration of backward pass

        ll = jnp.sum(jnp.log(c_scan))  # Store log-likelihood, log of Equation 13.63
        ll_norm = jnp.exp(ll / n_time_bins) # Normalize - where did this come from?
        #print("loop", perf_counter() - t0)

        ###################### POSTERIORS
        # Compute posterior distributions ######
        # Gamma - Equations 13.32, 13.64
        gammas = alphas_scan * betas_scan

        # Trials to compute xi
        trials_xi = np.arange(n_time_bins)
        # Exclude the first trial of every session
        trials_xi = trials_xi[~new_sess]

        # Equations 13.43 and 13.65
        # Xi summed across time steps
        xi_numer = ((alphas_scan.T[:, trials_xi - 1] /
                    c_scan[trials_xi]) @ 
                    (py_z.T[:, trials_xi] * 
                    betas_scan.T[:, trials_xi]).T
        )
        xis = xi_numer * transition_prob
        return gammas, xis, ll, ll_norm, alphas_scan, betas_scan

    def func_to_minimize(
        self,
        projection_weights, 
        n_features,
        n_states,
        y, 
        X, 
        gammas, 
    ):
        projection_weights = projection_weights.reshape(n_features, n_states)
        tmpy = self.observation_model.inverse_link_function(X @ projection_weights)
        nll = self.observation_model._negative_log_likelihood(
            y[:, jnp.newaxis],
            tmpy,
            aggregate_sample_scores = partial(lambda x: jnp.sum(gammas * x))
        )
        return nll

    def run_m_step(
        self,
        y: Array,
        X: Array,
        gammas: Array, 
        xis: Array, 
        projection_weights: Array, 
        new_sess: Array | None = None
    ):
        
        n_features = projection_weights.shape[0]
        n_states = projection_weights.shape[1]
        n_time_bins = X.shape[0]

        # Update Initial state probability eq. 13.18
        tmp_initial_prob = np.mean(gammas[:, new_sess], axis=1)
        initial_prob = tmp_initial_prob / np.sum(tmp_initial_prob)

        # Update Transition matrix eq. 13.19
        transition_prob = xis / np.sum(xis, axis=1)

        # Minimize negative log-likelihood to update GLM weights
        res = minimize(
            self.func_to_minimize, 
            projection_weights.flatten(),
            args = (
                n_features,
                n_states,
                y, 
                X, 
                gammas
            ) 
        )

        projection_weights = res.x

        return projection_weights
