---
jupytext:
  formats: ipynb,py:percent,md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# GLM-HMMs: The Learning Problem

Fitting a GLM-HMM means recovering its parameters (the initial state distribution, the transition matrix, and one GLM per state) from the observed choices alone, without seeing the hidden states that, we assume, produced them. Because the states are hidden, the likelihood of the observations is a sum over every possible sequence of states, and it has no closed-form maximum. This page explains why the fitting is problematic, how to fit this model in theory and practice, and what to watch out for when fitting real data. We will briefly cover:

- **The fitting problem.** How the model's assumptions factorize the joint distribution of states and choices, and why the resulting marginal likelihood is hard to maximize directly.
- **Expectation-maximization (EM).** The iterative algorithm used to fit GLM-HMMs, and the expected complete-data log-likelihood (ECLL) it maximizes at each iteration.
- **The E-step.** The forward-backward algorithm, which computes the posterior probabilities of the states.
- **The M-step.** Closed-form updates for the initial distribution and the transition matrix, a numerical fit for each state's GLM, and optional Dirichlet priors that regularize the probabilities.
- **Three practical caveats.** Why EM can converge to different local optima depending on where it starts (and how NeMoS initializes it), how to handle data recorded in separate sessions, and how to choose the number of states.
- **Fitting simulated data.** A fit that checks whether EM recovers the parameters used to generate the data.

```{code-cell} ipython3
:tags: [hide-input]

import warnings

# The helper functions live in nemos's documentation utilities, which warn on
# import that they are not part of the public API; silence that here.
warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)
```
```{code-cell} ipython3
# Imports
import nemos as nmo
import jax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Helper functions for this tutorial
from nemos import _documentation_utils as tutorial_utils
```
```{code-cell} ipython3
:tags: [hide-input]

# Enable 64-bit floating-point precision. Many quantities in a GLM-HMM (the
# forward-backward messages, products of small probabilities) are computed
# repeatedly and can become very small, so float64 guards against numerical
# error accumulating.
jax.config.update("jax_enable_x64", True)

# Parameters for plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.2, rc=custom_params)

# Seed a random number generator once here, so that every quantity sampled on
# this page comes out the same on each build of the docs.
rng = np.random.default_rng(2)
```

We use the session simulated at the end of [GLM-HMMs: An Introduction](README.md), in which choices switch between an engaged state and two biased states. The cell below regenerates it, so this page runs on its own.

```{code-cell} ipython3
:tags: [hide-input]

n_states = 3
T = 3000

# a single stimulus predictor (signed evidence) in [-1, 1]
X = rng.uniform(-1, 1, size=(T, 1))

# ground-truth parameters. coef_ has shape (n_features, n_states);
# the states are: 0 = engaged, 1 = biased-left, 2 = biased-right
true_coef = np.array([[5.0, 0.5, 0.5]])       # engaged: strong stimulus weight; biased: weak
true_intercept = np.array([0.0, 2.5, -2.5])   # engaged: no bias; biased: strong +/- bias
true_transition = np.array([[0.94, 0.03, 0.03],
                            [0.05, 0.90, 0.05],
                            [0.05, 0.05, 0.90]])  # every state is sticky
true_initial = np.array([0.85, 0.10, 0.05])       # usually start engaged

# a GLM-HMM whose parameters we set ourselves instead of fitting them
true_model = nmo.glm_hmm.GLMHMM(n_states=n_states, observation_model="Bernoulli")
true_model.coef_ = true_coef
true_model.intercept_ = true_intercept
true_model.transition_prob_ = true_transition
true_model.initial_prob_ = true_initial
true_model.scale_ = np.ones(n_states)  # the Bernoulli has no free scale parameter

# `simulate` walks the chain and emits one choice per trial: z_1 comes from the
# initial distribution, each z_t from the row of the transition matrix that the
# previous state selects, and each choice from the Bernoulli GLM of the state
# active on that trial. It also returns that state's P(choosing left) per trial.
y, prob_left, z_true = true_model.simulate(
    jax.random.PRNGKey(0), X, state_format="index"
)
```

## The fitting problem
A GLM-HMM is fully defined by the parameters $\boldsymbol{\theta} \equiv \{\boldsymbol{\pi},\boldsymbol{A},\{\boldsymbol{w}_k, b_k\}_{k=1}^K\}$:

- the initial state distribution $\boldsymbol{\pi} \in \mathbb{R}^K$, holding the probability of starting in each state;
- the transition matrix $\boldsymbol{A} \in \mathbb{R}^{K \times K}$, whose entry $A_{jk} = p(z_t = k \mid z_{t-1} = j)$ is the probability of moving from state $j$ to state $k$, including $k = j$;
- the per-state weights $\boldsymbol{w}_k \in \mathbb{R}^M$ and intercepts $b_k \in \mathbb{R}$, one pair per state, which define that state's GLM.

Because of the Markov, stationarity and independence assumptions, the joint probability distribution over both latent states $\boldsymbol{z}$ and observed variables $\boldsymbol{y}$, given the inputs $\boldsymbol{x}$ and the parameters $\boldsymbol{\theta}$, is given by:

\begin{equation}
p(\boldsymbol{y}, \boldsymbol{z} \mid \boldsymbol{x},\boldsymbol{\theta})
= p(z_1\mid\boldsymbol{\pi})
  \left[\prod_{t=2}^T p(z_t \mid z_{t-1},\boldsymbol{A})\right]
  \prod_{t=1}^T p(y_t\mid z_t, \boldsymbol{x}_t, \boldsymbol{w}_{z_t}, b_{z_t})
\end{equation}

```{admonition} The Markov, stationarity and independence assumptions
:class: note
We can factorize the joint distribution of the states and observations into three factors thanks to three assumptions: Markov, stationarity and independence (Ibe, 2013) <span id="cite1"></span><a href="#ref1">[1]</a>. The Markov assumption, which states that the next state depends only on the current one, is what reduces the transition term to a single factor per step; models where the next state depends on several previous states are possible but harder to analyze. The stationarity assumption, which states that transition probabilities are constant, is what lets one fixed matrix $\boldsymbol{A}$ serve for the whole experiment. And the observation independence assumption, which states that each observation is independent of the others given its own state and input, is what turns the emissions into a product $\prod_t p(y_t \mid z_t, \boldsymbol{x}_t, \boldsymbol{w}_{z_t}, b_{z_t})$.
```

Given that we have a set of observations, we can determine the parameters of a GLM-HMM using maximum likelihood. The likelihood function is obtained from the joint distribution by marginalizing over the hidden states. 

```{math}
:label: eq_marginal_likelihood
\begin{split}
p(\boldsymbol{y}\mid\boldsymbol{x}, \boldsymbol{\theta})
&= \sum_{\boldsymbol{z}} p(\boldsymbol{y}, \boldsymbol{z} \mid \boldsymbol{x},\boldsymbol{\theta}) \\
&= \sum_{\boldsymbol{z}} p(z_1\mid\boldsymbol{\pi})
   \left[\prod_{t=2}^T p(z_t \mid z_{t-1},\boldsymbol{A})\right]
   \prod_{t=1}^T p(y_t\mid z_t, \boldsymbol{x}_t, \boldsymbol{w}_{z_t}, b_{z_t})
\end{split}
```

Directly maximizing this likelihood is difficult for two reasons. First, the joint distribution $p(\boldsymbol{y}, \boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta})$ does not factorize into independent per-trial terms, because each transition term $p(z_t \mid z_{t-1})$ couples a latent state to the previous one. Unlike in a mixture model, the sum over $\boldsymbol{z}$ therefore cannot be split into $T$ independent sums over each $z_t$. Evaluating it by brute force would mean enumerating all $K^T$ sequences of latent states, a number that grows exponentially with the number of trials $T$. Second, marginalizing over $\boldsymbol{z}$ makes the likelihood a sum, over all state sequences, of products of initial-state, transition and state-specific GLM emission probabilities. The log-likelihood is therefore the logarithm of a sum, so the three groups of parameters do not appear in separate terms, and setting the derivatives to zero yields no closed-form solution (for more details in the HMM case, see Chapter 13 of Bishop, 2006 <span id="cite2a"></span><a href="#ref2a">[2a]</a>).


## The solution: expectation-maximization (EM)

EM, combined with the forward-backward algorithm, addresses both difficulties. The objective is to maximize a lower bound on the marginal log-likelihood. The way this is achieved is by iteratively making a guess about the latent state distribution and then using this updated guess to estimate the parameters.

- First, the E-step fixes the parameters at their values from the previous iteration, $\boldsymbol{\theta}^{\text{old}}$, and computes the posterior over the latent states, $p(\boldsymbol{z} \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})$. Only two sets of its marginals are needed: the single-trial posteriors $\gamma_{t,k}$ and the consecutive-pair posteriors $\xi_{t,j,k}$.
- Then, the M-step holds these posteriors fixed and maximizes the expected complete-data log-likelihood (ECLL, defined below), which separates into three terms: $\boldsymbol{\pi}$ is set to the state posteriors on the first trial, $\boldsymbol{A}$ to the expected transition counts normalized so that each row sums to one, and each state's GLM is fit by maximizing its log-likelihood over all trials weighted by $\gamma_{t,k}$, which generally requires numerical optimization. 

Each iteration of these two steps is guaranteed not to decrease the marginal log-likelihood $\log p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta})$, so the two steps are repeated until it stops improving, converging to a local, not necessarily global, maximum.

## The expected complete-data log-likelihood (ECLL)

During the E-step, we form the ECLL, which is a lower bound on the log-likelihood of the data given the parameters, using the current values of the parameters $\boldsymbol{\theta}^{\text{old}}$:

\begin{equation}
\text{ECLL}(\boldsymbol{\theta}) \equiv
\sum_{\boldsymbol{z}} p\!\left(\boldsymbol{z} \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}\right)
\log p\!\left(\boldsymbol{y}, \boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta}\right)
\end{equation}

We write $\gamma_{t,k}$ for the marginal posterior probability that trial $t$ was in state $k$, and $\xi_{t,j,k}$ for the joint posterior over two consecutive latents:

\begin{align}
\gamma_{t,k} &= p(z_t = k \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}) \\
\xi_{t,j,k} &= p(z_{t-1} = j, z_t = k \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})
\end{align}

Substituting the GLM-HMM joint distribution into the ECLL and collecting terms with these definitions, we find that the ECLL separates into three pieces that correspond exactly to the three objects of the model (the initial distribution, the transitions, and the per-state emissions):

```{math}
:label: eq_ecll_separated
\begin{split}
\text{ECLL}(\boldsymbol{\theta})
&= \sum_{k=1}^K \gamma_{1,k} \log \pi_k
 + \sum_{t=2}^T \sum_{j=1}^K \sum_{k=1}^K \xi_{t,j,k}\log A_{jk} \\
&\quad + \sum_{t=1}^T \sum_{k=1}^K \gamma_{t,k} \log p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}_k, b_k)
\end{split}
```

That separation is what makes the M-step straightforward. But how do we compute $\gamma$ and $\xi$? With the E-step.

```{admonition} How is it guaranteed never to decrease the marginal likelihood?
:class: note dropdown
For the derivation, see Bishop (2006), Section 9.4 <span id="cite2b"></span><a href="#ref2b">[2b]</a>, and Pillow (2024) <span id="cite3"></span><a href="#ref3">[3]</a>.

```


## The E-step: forward-backward algorithm
The E-step does not need the full posterior over all $K^T$ state sequences, only two sets of marginals:
- $\gamma_{t,k} = p(z_t = k \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})$, the probability of state $k$ on trial $t$, and
- $\xi_{t,j,k} = p(z_{t-1} = j, z_t = k \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})$, the probability of a transition from state $j$ to state $k$ between trials $t-1$ and $t$.

All quantities in the E-step are evaluated at the current parameter values $\boldsymbol{\theta}^{\text{old}} = \{\boldsymbol{\pi}^{\text{old}}, \boldsymbol{A}^{\text{old}}, \boldsymbol{w}^{\text{old}}_1, \dots, \boldsymbol{w}^{\text{old}}_K, b^{\text{old}}_1, \dots, b^{\text{old}}_K\}$. The forward-backward algorithm computes both sets of marginals in two recursive passes over the trials (Bishop, 2006, Eqs. 13.36 to 13.38) <span id="cite2c"></span><a href="#ref2c">[2c]</a>:

- **Forward pass.** $\alpha_{t,k} = p(y_1, \dots, y_t, z_t = k \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})$ is the joint probability of the observations up to and including trial $t$ and of being in state $k$ on trial $t$. 
- **Backward pass.** $\beta_{t,k} = p(y_{t+1}, \dots, y_T \mid z_t = k, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})$ is the probability of the choices after trial $t$ given state $k$ on trial $t$. 

Combining the two passes gives 
- $\gamma_{t,k} = \alpha_{t,k} \beta_{t,k} / p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})$ and 
- $\xi_{t,j,k} = \alpha_{t-1,j} A^{\text{old}}_{jk} \, p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}^{\text{old}}_k, b^{\text{old}}_k) \, \beta_{t,k} / p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})$. 

```{admonition} Why the pairwise posterior takes this form
:class: dropdown

By the definition of conditional probability,

$$\xi_{t,j,k} = \frac{p(z_{t-1} = j, z_t = k, \boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})}{p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}})}.$$

Splitting the choices into past ($y_{1:t-1}$), present ($y_t$) and future ($y_{t+1:T}$) and applying the product rule in time order, the numerator becomes a product of four terms, each simplified by the conditional independence structure of the HMM:

- $p(y_{1:t-1}, z_{t-1} = j \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}) = \alpha_{t-1,j}$, by definition;
- $p(z_t = k \mid z_{t-1} = j, y_{1:t-1}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}) = A^{\text{old}}_{jk}$, since the next state depends only on the current state;
- $p(y_t \mid z_t = k, z_{t-1} = j, y_{1:t-1}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}) = p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}^{\text{old}}_k, b^{\text{old}}_k)$, since the choice on trial $t$ depends only on the state and input on trial $t$;
- $p(y_{t+1:T} \mid z_t = k, y_t, z_{t-1} = j, y_{1:t-1}, \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}) = \beta_{t,k}$, since future choices depend on the past only through $z_t$.

Intuitively, the numerator is the total probability of all state sequences that pass through state $j$ on trial $t-1$ and state $k$ on trial $t$: $\alpha_{t-1,j}$ covers every way of reaching $j$, $A^{\text{old}}_{jk} \, p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}^{\text{old}}_k, b^{\text{old}}_k)$ is the step from $j$ to $k$, and $\beta_{t,k}$ covers every way of continuing from $k$ (Bishop, 2006, Eq. 13.43) <span id="cite2d"></span><a href="#ref2d">[2d]</a>.
```

In the M-step, the $\gamma$'s are used to update $\boldsymbol{\pi}$ and to weight each trial in each state's GLM fit, and the $\xi$'s are used to update $\boldsymbol{A}$.

```{admonition} The marginal likelihood is not only an internal quantity
:class: tip

Summing the forward variables over states on the last trial gives the marginal likelihood at the current parameters, $p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}) = \sum_k \alpha_{T,k}$. NeMoS returns its logarithm, summed over trials, from {meth}`GLMHMM.score <nemos.glm_hmm.GLMHMM.score>`, and we use it below to choose between fits started from different initializations.

```


```{admonition} The smoothed posterior is not only an internal quantity
:class: tip
$\gamma_{t,k}$ is computed here as a means to the M-step, but it is also a readout. NeMoS makes it available by calling {meth}`GLMHMM.smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>`. It is one of the two ways of asking which strategy was active on a given trial, taken up on [the decoding problem](plot_02_decoding_problem.md) page.
```

## The M-step: parameter updates
In the M-step, after having run the forward-backward algorithm, we maximize the ECLL with respect to the GLM-HMM parameters $\boldsymbol{\theta}$. Because the ECLL {eq}`eq_ecll_separated` splits into three independent terms, each parameter can be optimized separately. For the initial distribution $\boldsymbol{\pi}$ and the transition matrix $\boldsymbol{A}$, this yields closed-form updates:

\begin{align}
    \pi_k^\text{new}
    & = \frac{\gamma_{1,k}}{\sum_{j=1}^K\gamma_{1,j}} \\
     A_{jk}^\text{new} & = \frac{\sum_{t=2}^T \xi_{t,j,k}}{\sum_{t=2}^T\sum_{k'=1}^K \xi_{t,j,k'}}
\end{align}

Intuitively, the new initial distribution is just the (normalized) posterior over the first state, and the new transition probability from $j$ to $k$ is the expected number of $j\!\to\!k$ transitions divided by the expected number of transitions out of $j$, including those from $j$ back to itself. The third term is the emission: the GLM weights and intercepts $\{\boldsymbol{w}_k, b_k\}$ have no closed form and are updated numerically, by gradient-based optimization of the weighted GLM log-likelihood, each trial weighted by its responsibility $\gamma_{t,k}$. By default, NeMoS adds a Ridge penalty of strength 1.0 on the GLM coefficients to this objective (the `regularizer` and `regularizer_strength` arguments of {class}`GLMHMM <nemos.glm_hmm.GLMHMM>`), so the default update is a penalized fit; pass `regularizer="UnRegularized"` to get the maximum-likelihood update described here.

```{admonition} Dirichlet priors on the initial and transition probabilities
:class: dropdown

The closed-form updates above are maximum-likelihood estimates, which can be poorly behaved when a state or a transition is rarely visited. A Dirichlet prior on the initial and transition probabilities regularizes this. The Dirichlet is the natural prior over a categorical distribution (each row of $\boldsymbol{A}$, and $\boldsymbol{\pi}$, is one such distribution), and with it the M-step computes *maximum a posteriori* (MAP) rather than maximum-likelihood estimates. In NeMoS you set it through the `dirichlet_initial_proba` and `dirichlet_transition_proba` arguments of {class}`GLMHMM <nemos.glm_hmm.GLMHMM>`, each a vector or matrix of concentration parameters $\kappa \geq 1$, which enter the two updates as extra counts:

\begin{align}
    \pi_k^\text{new}
    & = \frac{\gamma_{1,k} + \kappa^{\boldsymbol{\pi}}_k - 1}{\sum_{j=1}^K\left(\gamma_{1,j} + \kappa^{\boldsymbol{\pi}}_j - 1\right)} \\
    A_{jk}^\text{new}
    & = \frac{\sum_{t=2}^T \xi_{t,j,k} + \kappa^{\boldsymbol{A}}_{jk} - 1}{\sum_{k'=1}^K\left(\sum_{t=2}^T \xi_{t,j,k'} + \kappa^{\boldsymbol{A}}_{jk'} - 1\right)}
\end{align}

Each $\kappa - 1$ acts as a pseudo-count added before any data is seen. If every $\kappa > 1$, every numerator is positive, so no entry can reach exactly 0 and therefore none can reach exactly 1; larger values pull the estimates more strongly toward the prior, and a large diagonal $\kappa^{\boldsymbol{A}}_{jj}$ encodes a preference for sticky states. Setting every $\kappa = 1$ adds nothing and recovers the maximum-likelihood updates; this flat prior is the NeMoS default (used when the arguments are left `None`). The GLM coefficients are regularized separately, through the `regularizer` argument described above, just as for a plain GLM.

```

## A caveat: the likelihood is not concave

EM is guaranteed never to decrease the marginal log-likelihood, but that does not mean it finds the best possible solution. The GLM-HMM marginal log-likelihood is not concave: it has multiple local optima, and once EM starts in the basin of attraction of one of them it cannot escape, so the result depends on the initialization (Salakhutdinov et al., 2003) <span id="cite4a"></span><a href="#ref4a">[4a]</a>. This holds even though each state's Bernoulli GLM has a concave log-likelihood. 

To see why, start from the case where the states are known. For a given state sequence $\boldsymbol{z}$, the log-likelihood is the logarithm of the joint distribution

\begin{equation}
\log p(\boldsymbol{y}, \boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta})
= \log p(z_1\mid\boldsymbol{\pi})
+ \sum_{t=2}^T \log p(z_t \mid z_{t-1}, \boldsymbol{A})
+ \sum_{t=1}^T \log p(y_t\mid z_t, \boldsymbol{x}_t, \boldsymbol{w}_{z_t}, b_{z_t}),
\end{equation}

which is a sum of concave terms (the logarithms of the initial and transition probabilities, and the Bernoulli GLM log-likelihoods), and hence concave. However, since they are unknown, the likelihood is instead a sum over all possible state sequences, see equation {eq}`eq_marginal_likelihood`. Each term in that sum has a single peak, located at the GLM weights that best fit that particular assignment of trials to states, and adding peaks located in different places can produce a surface with several peaks. 

This is also why EM is structured the way it is. The E-step freezes the assignment of trials to states as soft responsibilities, the posteriors $\gamma_{t,k}$ and $\xi_{t,j,k}$ computed at $\boldsymbol{\theta}^{\text{old}}$. With these held fixed, the ECLL separates into three terms, one for each group of parameters ({eq}`eq_ecll_separated`), and the M-step maximizes each term on its own. The initial-distribution and transition terms have the closed-form solutions given above. The emission term, $\sum_{t}\sum_{k}\gamma_{t,k}\log p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}_k, b_k)$, splits into one weighted GLM fit per state, and each is a weighted sum of concave GLM log-likelihoods, so it is concave too: any local maximum is the global one, and it can be found from any starting point. Each M-step is therefore a well-behaved concave problem. The difficulty lies in the sequence of them: every E-step recomputes the responsibilities from the current parameters, so each M-step solves a different problem, and the optimum EM eventually reaches depends on where it started.


The practical consequence is a heuristic: refit the model from several initializations and keep the solution with the highest log-likelihood, which is what we do below. In NeMoS, the `seed` argument of {class}`GLMHMM <nemos.glm_hmm.GLMHMM>` is a JAX PRNG key (by default `jax.random.PRNGKey(123)`) that sets the random draws made during initialization, while {meth}`GLMHMM.setup <nemos.glm_hmm.GLMHMM.setup>` chooses how each group of parameters is initialized (see the tip below). The default schemes already give EM a sensible starting point, and with them the seed affects only the GLM coefficients. Every coefficient is an independent draw from a zero-mean Gaussian with a standard deviation of 0.001, whatever the observation model. Each state's intercept is then set so that the model's predicted mean at initialization equals the mean of the observations. Because the coefficients start near zero, that prediction barely depends on the inputs, so the intercept alone decides it. The rule is the same for all observation models: take `y.mean()`, and apply the link function. What changes between observation models is only which link that is, and therefore what the resulting number means:

| Observation model | Default inverse link | Intercept initialized to |
|---|---|---|
| Bernoulli | logistic | $\log\!\big(\bar{y}/(1-\bar{y})\big)$ |
| Categorical | log-softmax | $\log$ of each class frequency, centered to sum to zero |
| Poisson | exp | $\log \bar{y}$ |
| Negative Binomial | exp | $\log \bar{y}$ |
| Gaussian | identity | $\bar{y}$ |
| Gamma | $1/x$ | $1/\bar{y}$ |

Before the link is applied, the mean is checked against the observation model's `rate_range`: $(0, 1)$ for Bernoulli, $(0, \infty)$ for Poisson, Gamma, Negative Binomial and Categorical, and $(-\infty, \infty)$ for Gaussian. A mean outside that range raises an error, and a mean exactly on a boundary (for example, a Bernoulli session with no 1s) is shifted inside by half an observation, with a warning.

The same intercept is used for all $K$ states, so the states start out identical except for their small random coefficients. `scale_` starts at 1.0 for every state; it matters only for models with a free scale (Gaussian, Gamma, Negative Binomial). The initial distribution starts uniform, and the transition matrix starts sticky, with a self-transition probability of 0.95 and the remaining 0.05 split evenly among the other states.

```{admonition} Choosing the initialization yourself
:class: tip
{meth}`GLMHMM.setup <nemos.glm_hmm.GLMHMM.setup>` selects how each group of parameters is initialized, either by name or with a callable of your own: `initial_proba_init` (`"uniform"`, `"random"`, `"dirichlet"`, `"kmeans"`), `transition_proba_init` (`"sticky"`, `"uniform"`, `"random"`, `"dirichlet"`, `"kmeans"`), `glm_params_init` (`"random"`, `"kmeans"`) and `scale_init` (`"constant"`, `"kmeans"`), each with a matching `..._kwargs` dictionary for its arguments. The `"kmeans"` options are worth knowing about here: they cluster the data first and start each state from its own cluster, which separates the states from the outset rather than relying on small random noise to pull them apart.

If you would rather skip initialization altogether, pass `init_params` to {meth}`fit <nemos.glm_hmm.GLMHMM.fit>`. It takes the tuple `(coef, intercept, scale, initial_prob, transition_prob)` and bypasses the initializers entirely, which is what you want when you are restarting from an earlier fit or testing whether EM recovers a known solution from a known starting point.
```

## A caveat: handling session breaks

Real experiments come in separate sessions recorded on different days, and we do not want the model to treat the last trial of one session and the first trial of the next as a genuine transition. This matters because the E-step computes the state probabilities recursively over trials: the forward pass carries what is known about the state from each trial to the next, and the backward pass carries it from each trial to the one before. The model as described so far assumes that all trials form a single chain, so if sessions were simply concatenated, both passes would carry information across the break, and the states inferred at the end of one session would shape those inferred at the start of the next, and vice versa.

When the data contain several sessions, we instead treat each session as its own chain: an independent draw from the same GLM-HMM, with all sessions sharing the same parameters. The E-step then runs the forward and backward recursions separately within each session. The forward pass restarts from the initial distribution $\boldsymbol{\pi}$ at the first trial of each session, the backward pass restarts from $\beta = 1$ at its last trial, and no transition is counted between sessions. The M-step pools the sessions: $\boldsymbol{\pi}$ is estimated from the first trial of every session, $\boldsymbol{A}$ from the transitions within sessions, and each state's GLM from all trials (Bishop, 2006) <span id="cite2e"></span><a href="#ref2e">[2e]</a>. The marginal log-likelihood is then the sum of the per-session log-likelihoods.

In NeMoS, session breaks are specified with the `session_starts` argument to `fit`, `score`, `smooth_proba`, `filter_proba`, `decode_state` and `simulate`. It accepts a boolean or 0/1 array of shape `(n_samples,)`, an integer array of session-start indices, or, when `X` or `y` is a pynapple `Tsd`/`TsdFrame`, a `pynapple.IntervalSet`. If `session_starts` is not given, pynapple inputs are split at the epochs of their `time_support`, and any other input is treated as a single chain.

## A caveat: how to choose a number of states
The number of hidden states $K$ is a hyperparameter: it must be fixed before fitting begins, and NeMoS requires it as the `n_states` argument. There are two separate reasons for this. First, $K$ determines the shape of everything we are fitting: $\boldsymbol{\pi}$ has $K$ entries, $\boldsymbol{A}$ is $K \times K$, and there is one weight vector and one intercept per state. EM searches a parameter space of fixed dimension, so that space does not exist until $K$ is pinned down. Second, the training likelihood cannot be used to choose $K$. A $(K+1)$-state model contains every $K$-state model as a special case: setting the initial probability of the extra state, and every transition probability into it, to zero means the extra state is never visited, which recovers the $K$-state model exactly. The best $(K+1)$-state fit is therefore always at least as good as the best $K$-state fit, and comparing training likelihoods would always favor the largest $K$. 

If it is not possible to use the training likelihood for assessing how many states to have, then how do we determine $K$? In practice, one fits models across a range of $K$ and compares them with a criterion that penalizes complexity (AIC, BIC) or measures generalization (held-out log-likelihood, cross-validation). These comparisons are worth pairing with a recovery analysis: simulate data from plausible parameters at the number of trials actually available, then check whether fitting recovers those parameters. 

A larger $K$ is not always better, nor always feasible, because the number of parameters grows while the data informing each one shrinks. The parameter count grows quadratically with $K$: $K-1$ for the initial distribution, $K(K-1)$ for the transition matrix, and $K(M+1)$ for the GLMs, with $M$ weights and one intercept per state. Meanwhile, every added state leaves fewer trials for the others. Each state's GLM parameters are learned only from the trials the chain spends in that state. Its row of $\boldsymbol{A}$ is learned from the same trials, but where the chain goes when it leaves is learned only from the trials on which it actually leaves, and these are few when the state is sticky.

Finally, statistical criteria are not the only consideration: choosing a $K$ also depends on what the states are for. In neuroscience, each state is meant to stand for something we can interpret. Larger $K$ makes this harder. Additional states tend to split an existing state into variants that differ only slightly, and each is estimated from fewer trials, so the states become harder to tell apart and to interpret. A smaller $K$ can therefore be preferable even when a larger one predicts held-out data slightly better. Ashwood et al. (2022) <span id="cite5"></span><a href="#ref5">[5]</a>, for example, found that held-out log-likelihood kept improving slightly with four and five states, but analyzed the three-state model for simplicity and interpretability, noting that the four-state model mainly split the engaged state into two substates that differed slightly in bias. If the goal is instead purely predictive, as in many machine-learning applications, the states need not be interpretable, and $K$ can be chosen by held-out performance alone.

(glm_hmm_fitting_simulated_data)=
## Fitting our simulated data
Now we can know how the fitting works! The theoretical explanation was important to understand what is under the hood of how easy it is to fit a GLM-HMM with NeMoS. Really, you only need to instantiate a {class}`GLMHMM <nemos.glm_hmm.GLMHMM>`, tell it how many states to look for, and call `fit`. In this short section we fit the model from six random initializations and keep the fit with the highest log-likelihood. We pass `regularizer="UnRegularized"` so that the GLM updates are the maximum-likelihood fits described above, instead of the default Ridge-penalized ones. We also use the true number of states, $K = 3$; with real data, you should choose $K$ by cross-validation.


```{code-cell} ipython3
# refit from several seeds and keep the highest-likelihood solution
best_ll, best_model = -np.inf, None
for s in range(6):
    candidate = nmo.glm_hmm.GLMHMM(
        n_states=n_states,
        regularizer="UnRegularized",
        seed=jax.random.PRNGKey(s),
    )
    candidate.fit(X, y)
    ll = float(candidate.score(X, y))
    if ll > best_ll:
        best_ll, best_model = ll, candidate

# EM's state labels are arbitrary: reorder the fitted states (engaged first,
# then biased-left, then biased-right) so they line up with the true states
best_model = tutorial_utils.relabel_by_bias(best_model)

fig = tutorial_utils.plot_param_recovery(
    true_coef, true_intercept,
    np.asarray(best_model.coef_), np.asarray(best_model.intercept_),
    true_transition, np.asarray(best_model.transition_prob_),
    state_labels=["engaged", "biased-left", "biased-right"],
)
```

EM has recovered the generative model: one engaged state with a large stimulus weight and near-zero bias, two biased states of opposite sign whose choices barely depend on the stimulus, and a transition matrix with a heavy diagonal, closely matching the one we simulated from in [GLM-HMMs: An Introduction](README.md).

## References
<a id="ref1"><a href="#cite1">[1]</a> [Ibe, O. C. (2013). Hidden Markov models. In Markov Processes for Stochastic Modeling (2nd ed., Chapter 14, pp. 417–451). Elsevier.](https://doi.org/10.1016/B978-0-12-407795-9.00014-1)

<a id="ref2a"><a href="#cite2a">[2a]</a> <a id="ref2b"><a href="#cite2b">[2b]</a> <a id="ref2c"><a href="#cite2c">[2c]</a> <a id="ref2d"><a href="#cite2d">[2d]</a> <a id="ref2e"><a href="#cite2e">[2e]</a> Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

<a id="ref3"><a href="#cite3">[3]</a> [Pillow, J. (2024). Lecture 18 notes: Expectation Maximization (EM). Statistical Modeling and Analysis of Neural Data (NEU 560), Princeton University.](https://pillowlab.princeton.edu/teaching/statneuro2024/notes/notes16_EM.pdf)

<a id="ref4a"><a href="#cite4a">[4a]</a> Salakhutdinov, R., Roweis, S. T., & Ghahramani, Z. (2003). Optimization with EM and expectation-conjugate-gradient. In Proceedings of the 20th International Conference on Machine Learning (ICML-03) (pp. 672–679).

<a id="ref5"><a href="#cite5">[5]</a> [Ashwood, Z. C., Roy, N. A., Stone, I. R., The International Brain Laboratory, Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201–212.](https://doi.org/10.1038/s41593-021-01007-z)
