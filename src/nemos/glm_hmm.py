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
            projection_weights: Array,
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
        transition_prob : .w
            (n_features x n_states) latent state GLM weights
        initial_prob : .pi
            (n_states x 1) initial latent state probability
        projection_weights : .A
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
        n_states = projection_weights.shape[1]  # number of latent states from dimensions of w

        # Revise if the data is one single session or multiple sessions. If new_sess is not provided, assume one session
        if new_sess is None:
            new_sess = np.zeros_like(y, dtype=bool)
        new_sess[0] = True

        # Firing rate
        tmpy = self.observation_model.inverse_link_function(projection_weights.T @ X.T)
        
        # Data likelihood p(y|z) from emissions model using NeMoS
        py_z = jnp.exp(
            self.observation_model.log_likelihood(
                y,
                tmpy,
               aggregate_sample_scores = partial(lambda x: x)
            )
        )   

        ###### Forward recursion to compute alphas ######
        # Initialize variables
        alphas = np.full((n_states, n_time_bins), np.nan) # forward pass alphas
        c = np.full(n_time_bins, np.nan) # variable to store marginal likelihood

        #py_z_slices = jax.tree_util.tree_map(
        #    lambda x: jnp.mo
        #)

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

    def run_m_step(
            gammas: Array,
            xis: Array,
            projection_weights: Array,
            new_sess: Array | None = None
            ):
        
        # Update Initial state probability eq. 13.18
        tmp_pi = np.mean(gammas[:, new_sess], axis=1)
        initial_prob = tmp_pi / np.sum(tmp_pi)

        # Update Transition matrix eq. 13.19
        transition_prob = xis / np.sum(xis, axis=1)

        # Minimize negative log-likelihood to update GLM weights
        projection_weights = minimize_likelihood()

        return projection_weights
                                                                                                                            
    def run_baum_welch_jax(
        self,
        X: Array,
        y: Array,
        initial_prob: Array,
        transition_prob: Array,
        projection_weights: Array,
        new_sess: Array | None = None,
    ):
        """ "
        Baum-Welch algorithm to compute the forward-backward algorithm and return the marginal posterior distribution.

        According to Bishop's "Pattern Recognition and Machine Learning".

        Parameters
        ----------
        X :
            (n_time_bins x n_features) design matrix
        y :
            (1 x n_time_bins) observations
        model :
            GLM-HMM model object containing the parameters
        .w :
            (n_features x n_states) latent state GLM weights
        .pi :
            (n_states x 1) initial latent state probability
        .A :
            (n_states x n_states) latent state transition matrix
        new_sess :
            logical array with 1s denoting the start of a new session.
            If unspecified or empty, treats the full set of trials as a single session.

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
        n_states = projection_weights.shape[
            1
        ]  # number of latent states from dimensions of w

        # Revise if the data is one single session or multiple sessions. If new_sess is not provided, assume one session
        if new_sess is None:
            new_sess = jnp.zeros_like(y, dtype=bool)
        new_sess[0] = True

        # Data likelihood p(y|z) from emissions model
        # Compute likelihood given the fixed weights
        # This would be outputted by nemos I believe although there is no
        # Fitting to be done here - i still it should be done by nemos so
        # there is no hard coding of the observation model

        # okay will just to as sara did and then use a function to compute the likelihood later
        tmpy = 1 / (
            1 + jnp.exp(-projection_weights.T @ X.T)
        )  # f(projection_weights, x[:,0])
        py_z = y * tmpy + (1 - y) * (1 - tmpy)  # p(y|z)
        # py_z = compute_likelihood()
        print("SHAPE", py_z.shape)
        print("SHAPE proj", projection_weights.shape)
        print("SHAPE X", X.shape)

        # Forward recursion to compute alphas ######
        # Initialize variables
        alphas = np.full((n_states, n_time_bins), np.nan)  # forward pass alphas
        c = np.full(n_time_bins, np.nan)  # variable to store marginal likelihood

        py_z_slices = jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, -1, 0), py_z) # No need anymore because dimensions are correct

        new_sess = jnp.asarray(new_sess)
        initial_prob = jnp.asarray(initial_prob)

        with jax.disable_jit(False):
            alphas_scan, c_scan = forward_pass(
                initial_prob, transition_prob, py_z_slices, new_sess
            )  # these are equivalent to the forward pass with python loop

        t0 = perf_counter()
        with jax.disable_jit(False):
            betas_scan = backward_pass(transition_prob, py_z_slices, c_scan, new_sess)
        print("\nscan", perf_counter() - t0)

        for t in range(n_time_bins):
            if new_sess[t]:
                alphas[:, t] = (
                    initial_prob * py_z[:, t]
                )  # Initial alpha. Equation 13.37. Reinitialize for new sessions
            else:
                alphas[:, t] = py_z[:, t] * (
                    transition_prob.T @ alphas[:, t - 1]
                )  # Equation 13.36

            c[t] = np.sum(alphas[:, t])  # Store marginal likelihood
            if (
                c[t] == 0
            ):  # This should not happen, but if it does, raise an error if weights are out of control
                raise ValueError(
                    f"Zero marginal likelihood at time {t} - Weights may be out of control"
                )
            alphas[:, t] /= c[t]  # Normalize (Equation 13.59)

        ll = np.sum(np.log(c))  # Store log-likelihood
        ll_norm = np.exp(ll / n_time_bins)

        # Backward recursion to compute betas
        # Initialize variables
        betas = np.full((n_states, n_time_bins), np.nan)  # backward pass betas
        betas[:, -1] = np.ones(n_states)  # initial beta (Equation 13.39)

        # Solve for remaining betas
        t0 = perf_counter()
        for t in range(n_time_bins - 2, -1, -1):
            if new_sess[t + 1]:
                betas[:, t] = np.ones(
                    n_states
                )  # Reinitialize backward pass if end of session
            else:
                betas[:, t] = transition_prob @ (
                    betas[:, t + 1] * py_z[:, t + 1]
                )  # Equation 13.38
                betas[:, t] /= c[t + 1]  # Normalize (Equation 13.62)
        print("loop", perf_counter() - t0)
        print("max abs diff", np.abs(betas_scan - betas.T).max())
        # Compute posterior distributions ######
        gammas = alphas * betas  # Gamma - Equations 13.32, 13.64

        # Trials to compute xi
        # Exclude the first trial of every session
        # Transition matrix
        trials_xi = np.arange(n_time_bins)
        trials_xi = trials_xi[~new_sess]

        # Equations 13.43 and 13.65
        # Xi summed across time steps
        xi_numer = (alphas[:, trials_xi - 1] / c[trials_xi]) @ (
            py_z[:, trials_xi] * betas[:, trials_xi]
        ).T
        xis = xi_numer * transition_prob
        # print(xi_numer * transition_prob)
        return gammas, xis, ll, ll_norm, alphas, betas


    def run_m_step(
        gammas: Array, xis: Array, projection_weights: Array, new_sess: Array | None = None
    ):

        # Update Initial state probability eq. 13.18
        tmp_pi = np.mean(gammas[:, new_sess], axis=1)
        initial_prob = tmp_pi / np.sum(tmp_pi)

        # Update Transition matrix eq. 13.19
        transition_prob = xis / np.sum(xis, axis=1)

        # Minimize negative log-likelihood to update GLM weights
        projection_weights = minimize_likelihood()

        return projection_weights
