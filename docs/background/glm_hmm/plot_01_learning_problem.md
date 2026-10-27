---
jupytext:
  formats: ipynb,py:percent,md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: nemos (3.13.3.final.0)
  language: python
  name: python3
---
# GLM-HMMs: The Learning Problem

Fitting a GLM-HMM means recovering its parameters from observed choices alone, without ever seeing the states that produced them. This page works through how that is done, using the session simulated at the end of [GLM-HMMs: An Introduction](README.md); the cell below regenerates it, so the page runs on its own. One thing to expect from the outset: the likelihood we maximize is not convex, so a single fit can settle on a poor solution, and fitting in practice means running EM from several initializations and keeping the best of them — see {ref}`glmhmm_non_convexity`. Reading the states back out of a fitted model is taken up in [GLM-HMMs: The Decoding Problem](plot_02_decoding_problem.md).

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

The data we fit here is the session simulated at the end of [GLM-HMMs: an overview](README.md). The cell below regenerates it, so this page runs on its own.

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
y = np.asarray(y, dtype=float)
z_true = np.asarray(z_true, dtype=int)
```

## The fitting problem

When fitting a GLM-HMM, the goal is to learn the set of parameters $\boldsymbol{\theta} \equiv \{A,\boldsymbol{\pi},\{\boldsymbol{w}_k, b_k\}_{k=1}^K\}$ — the transition matrix $\boldsymbol{A} \in \mathbb{R}^{K \times K}$, the initial state distribution $\boldsymbol{\pi} \in \mathbb{R}^K$, and the per-state weights and intercepts $\{\boldsymbol{w}_k, b_k\}_{k=1}^K$ with $\boldsymbol{w}_k \in \mathbb{R}^M$ and $b_k \in \mathbb{R}$ — that maximize the likelihood of the observed data. Because the states are hidden, the likelihood of the observations alone is obtained by marginalizing the joint distribution over every possible state sequence:

\begin{equation}
p(\boldsymbol{y}\mid\boldsymbol{x}, \boldsymbol{\theta})
= \sum_{\boldsymbol{z}} p(\boldsymbol{y}, \boldsymbol{z} \mid \boldsymbol{x},\boldsymbol{\theta})
= \sum_{\boldsymbol{z}} p(z_1\mid\boldsymbol{\pi}) \left[\prod_{t=2}^T p(z_t \mid z_{t-1},A)\right]\prod_{t=1}^T p(y_t\mid z_t, \boldsymbol{w}_{z_t})
\end{equation}

Directly maximizing this is hard, because the sum over all possible state sequences $\boldsymbol{z}$ couples the parameters together. The standard solution is the Expectation-Maximization (EM) algorithm.

### EM in one paragraph
The difficulty is circular: if we knew which state was active on each trial, updating the parameters would be easy (each state's GLM would be fit only on its own trials, and $\boldsymbol{A}$ and $\boldsymbol{\pi}$ would be read off from the state sequence); and if we knew the parameters, computing the distribution over states would be easy. EM breaks the circle by alternating between the two:

- **E-step:** given the current parameters $\theta^{\text{old}}$, compute the posterior distribution over the hidden states using the forward-backward algorithm (below). This yields "responsibilities" — how much each trial belongs to each state.
- **M-step:** given those responsibilities, update the parameters $\theta$ to increase the *expected complete-data log-likelihood* (ECLL).

We iterate until the log-likelihood plateaus. Each full E-then-M iteration is guaranteed not to *decrease* the marginal log-likelihood of the data, so EM climbs steadily to a (local) maximum. With that intuition in place, we now make the E- and M-steps precise.

### The expected complete-data log-likelihood
During the E-step, we form the ECLL, which is a lower bound on the log-likelihood of the data given the parameters, using some current setting of the parameters $\theta^{\text{old}}$:

\begin{equation}
\text{ECLL}(\theta) = \sum_z p(\textbf{z}\mid \boldsymbol{y}, \{\boldsymbol{x}\}^T_{t=1},\theta^{\text{old}}) \log p(\textbf{y}, \textbf{z}\mid \theta, \{\boldsymbol{x}\}^T_{t=1}) \\
\end{equation}

Two posterior quantities do all the work here. We write $\gamma_{t,k}$ for the marginal posterior probability that trial $t$ was in state $k$, and $\xi_{t,j,k}$ for the joint posterior over two consecutive latents:

\begin{align}
\gamma_{t,k} &= p(z_t = k \mid \boldsymbol{y}, \{\boldsymbol{x}\}^T_{t=1}, \theta^{\text{old}}) \\
\xi_{t,j,k} &= p(z_t = k, z_{t-1} = j \mid \boldsymbol{y}, \{\boldsymbol{x}\}^T_{t=1}, \theta^{\text{old}})
\end{align}

Substituting the GLM-HMM joint distribution into the ECLL and collecting terms with these definitions, the ECLL separates into three pieces that correspond exactly to the three objects of the model — the initial distribution, the transitions, and the per-state emissions:

```{math}
:label: eq_ecll_separated
\begin{split}
\text{ECLL}(\theta) = \sum_{k=1}^K \gamma_{1,k} \log \boldsymbol{\pi}_k + \sum_{t=1}^T \sum_{j=1}^K \sum_{k=1}^K \xi_{t,j,k}\log A_{jk} \\
+ \sum_{t=1}^T \sum_{k=1}^K \gamma_{t,k} \log p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}_k)
\end{split}
```

That separation is what makes the M-step easy. But it leaves us needing $\gamma$ and $\xi$, which is what the forward-backward algorithm provides.

### The E-step: forward-backward algorithm
The E-step needs the posterior probabilities of the hidden states given *all* the data and the current parameters. The forward-backward algorithm computes them efficiently, in two passes over the sequence:

- The **forward pass** sweeps left to right and accumulates evidence *from the past*: after trial $t$ it holds, for each state $k$, the joint probability $\alpha_{t,k}$ of everything observed up to and including trial $t$ and of being in state $k$ at trial $t$.
- The **backward pass** sweeps right to left and accumulates evidence *from the future*: $\beta_{t,k}$ is the probability of everything observed *after* trial $t$, given that trial $t$ was in state $k$.
- **Combining** the two at each trial gives the smoothed posterior $\gamma_{t,k} \propto \alpha_{t,k}\beta_{t,k}$: the probability that the system was in state $k$ at trial $t$ given the *whole* observed sequence, past and future together. This is precisely the quantity the M-step needs in order to weight each trial's contribution to each state's update.

Both passes are recursive — each step reuses the previous one — which is what turns a sum over exponentially many state sequences into a computation linear in the number of trials. A useful by-product is that the forward pass hands us the data likelihood for free: $\sum_k \alpha_{T,k}$ is the marginal likelihood of the whole sequence, the very quantity EM is monotonically increasing.

:::{dropdown} The forward and backward recursions in full
The forward quantity is the joint probability of the activity up until trial $t$ and the latent state at trial $t$ being state $k$,

\begin{equation}
\alpha_{t,k} \equiv p(\boldsymbol{y}_{1:t}, z_t \mid \{\boldsymbol{x}_{t'}\}_{t'=1}^t)
\end{equation}

The probability associated with trial 1 is

\begin{equation}
\alpha_{1,k} = \boldsymbol{\pi}_k\, p(\boldsymbol{y}_{1} \mid z_1,\boldsymbol{x}_1, \boldsymbol{w}_k)
\end{equation}

where, in our case, $p(\boldsymbol{y}_{1} \mid z_1,\boldsymbol{x}_1, \boldsymbol{w}_k)$ is the usual Bernoulli GLM distribution mentioned above. For trials $1<t \leq T$, we obtain the probabilities recursively:

\begin{equation}
\alpha_{t,k} = \sum_{j=1}^K \alpha_{t-1,j}\,A_{j,k}\,p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}_k)
\end{equation}

During the backward pass, the goal is to calculate the probability of the choice data beyond the current trial,

\begin{equation}
\beta_{t,k} \equiv p(\boldsymbol{y}_{[t+1:T]}\mid z_t = k, \{\boldsymbol{x}_{t'}\}^T_{t'=t+1})
\end{equation}

These quantities are calculated recognizing that $\beta_{T,k} = 1$ and, for $t \in \{T-1, ...,1\}$:

\begin{equation}
\beta_{t,j} = \sum_{k=1}^K\beta_{t+1,k}\,A_{jk}\,p(y_{t+1}\mid z_{t+1} = k, \boldsymbol{x}_{t+1},\boldsymbol{w}_k)
\end{equation}

From the $\alpha_{t,k}$ and $\beta_{t,k}$ quantities we can then form the single and joint posterior state probabilities needed to maximize the ECLL:

\begin{align}
\gamma_{t,k}
    &= p(z_t = k \mid D, \theta^{\text{old}})\\
    &= \frac{p(y_{[1:t]}, z_t = k \mid \{\boldsymbol{x}\}_{t'=1}^t, \theta^{\text{old}})\, p(y_{[t+1:T]} \mid z_t = k, \{\boldsymbol{x}\}_{t'=t+1}^T, \theta^{\text{old}})}{p(\boldsymbol{y} \mid \{\boldsymbol{x}\}_{t'=1}^T,\theta^{\text{old}})} \\
    &= \frac{\alpha_{t,k}\beta_{t,k}}{\sum_{k=1}^K\alpha_{T,k}}
\end{align}

and similarly for $\xi$:

\begin{equation}
    \xi_{t,j,k} = \frac{\alpha_{t,j}\,A_{j,k}\,\beta_{t+1,k}\,p(y_{t+1}\mid z_{t+1} = k, \boldsymbol{x}_{t+1}, \boldsymbol{w}_k)}{\sum_{k=1}^K\alpha_{T,k}}
\end{equation}

where $p(y_{t+1}\mid z_{t+1} = k, \boldsymbol{x}_{t+1}, \boldsymbol{w}_k)$ is again the Bernoulli GLM distribution.
:::

That marginal likelihood has a name of its own: computing how probable the observed sequence is under a given model is the third classical HMM problem, the **evaluation problem**, and the forward pass answers it in passing rather than needing an algorithm of its own. It is what {meth}`GLMHMM.score <nemos.glm_hmm.GLMHMM.score>` returns, what EM watches to decide it has converged, and what we use below to choose between fits started from different initializations.

```{admonition} The smoothed posterior is not only an internal quantity
:class: tip
$\gamma_{t,k}$ is computed here as a means to the M-step, but it is also a readout in its own right, and NeMoS exposes it as {meth}`GLMHMM.smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>`. It is one of the two ways of asking which strategy was active on a given trial, taken up on [the decoding problem](plot_02_decoding_problem.md) page.
```

```{admonition} Handling session breaks
:class: note
Real experiments come in separate sessions recorded on different days, and we do not want the model to treat the last trial of one session and the first trial of the next as a genuine transition. NeMoS handles this through the `session_starts` argument to `fit`, `smooth_proba`, `decode_state`, and `simulate`: the **transition** term is dropped across session boundaries (so the state chain is reset at each session start), while the **emission** term still applies to every trial. Concretely, `session_starts` accepts a boolean/0-1 array of shape `(n_samples,)`, an integer array of session-start indices, or — when `X` or `y` is a pynapple `Tsd`/`TsdFrame` — a `pynapple.IntervalSet`.

Once `session_starts` is supplied, each session is a chain of its own, as opposed to all sessions being one chain together. This distinction matters for history predictors. The previous choice, for instance, is undefined on the first trial of a session, since the preceding trial belongs to a different day; marking those entries `NaN` is allowed precisely because they sit at a chain's first trial. As such, be aware of carrying the last choice of one session over to the first trial of the next, which would invent a dependency between trials recorded days apart. Our example here is a single continuous session, so we omit `session_starts`.
```

### The M-step: parameter updates
In the M-step, after having run the forward-backward algorithm, we maximize the ECLL with respect to the GLM-HMM parameters $\theta$. Because the ECLL {eq}`eq_ecll_separated` splits into three independent terms, each parameter can be optimized separately. For the initial distribution $\boldsymbol{\pi}$ and the transition matrix $\boldsymbol{A}$, this yields closed-form updates:

\begin{align}
    \pi_k^\text{new}
    & = \frac{\gamma_{1,k}}{\sum_{j=1}^K\gamma_{1,j}} \\
     A_{jk}^\text{new} & = \frac{\sum_{t=2}^T \xi_{t,j,k}}{\sum_{t=2}^T\sum_{k=1}^K \xi_{t,j,k} }
\end{align}

Intuitively, the new initial distribution is just the (normalized) posterior over the first state, and the new transition probability from $j$ to $k$ is the expected number of $j\!\to\!k$ transitions divided by the expected number of times we left $j$. The third term is the emission: the GLM weights $\{\boldsymbol{w}_k\}$ have no closed form and are updated numerically, by gradient-based optimization of the weighted GLM log-likelihood, each trial weighted by its responsibility $\gamma_{t,k}$.

```{admonition} Dirichlet priors on the initial and transition probabilities
:class: note
The closed-form updates above are maximum-likelihood estimates, which can be poorly behaved when a state or a transition is rarely visited — an entry can be driven all the way to 0 or 1. A **Dirichlet prior** on the initial and transition probabilities regularizes this. The Dirichlet is the natural prior over a categorical distribution (each row of $\boldsymbol{A}$, and $\boldsymbol{\pi}$, is one such distribution): its concentration parameters act like pseudo-counts, keeping every entry a valid probability and pulling the estimates away from degenerate 0-or-1 rows and, if desired, toward stickiness. In NeMoS this turns the M-step updates into *maximum a posteriori* updates, and you control it through the `dirichlet_initial_proba` and `dirichlet_transition_proba` arguments of {class}`GLMHMM <nemos.glm_hmm.GLMHMM>` (each a vector/matrix of Dirichlet concentration parameters; left `None`, a flat, uninformative prior is used). Concretely, the concentration parameters enter the two updates above as extra counts, and the maximum-likelihood estimates become maximum a posteriori ones:

\begin{align}
    \pi_k^\text{new}
    & = \frac{\gamma_{1,k} + \alpha^{\boldsymbol{\pi}}_k - 1}{\sum_{j=1}^K\left(\gamma_{1,j} + \alpha^{\boldsymbol{\pi}}_j - 1\right)} \\
    A_{jk}^\text{new}
    & = \frac{\sum_{t=2}^T \xi_{t,j,k} + \alpha^{\boldsymbol{A}}_{jk} - 1}{\sum_{k'=1}^K\left(\sum_{t=2}^T \xi_{t,j,k'} + \alpha^{\boldsymbol{A}}_{jk'} - 1\right)}
\end{align}

Each $\alpha - 1$ is the pseudo-count the prior contributes before any data is seen, which is why a rarely visited state or transition can no longer be pushed to exactly zero. Setting every $\alpha = 1$ contributes nothing and recovers the maximum-likelihood updates above, which is the flat prior NeMoS uses by default; values below one are not currently supported. The GLM weights, in turn, are regularized through the `regularizer` argument, just as for a plain GLM.
```

(glmhmm_non_convexity)=
### A caveat: the likelihood is non-convex
EM is guaranteed to increase (or hold) the marginal log-likelihood at every iteration, but that does *not* mean it finds the best possible solution. The GLM-HMM marginal likelihood is **non-convex**: it has multiple local optima, and EM converges to whichever one lies uphill from where it started, so the result depends on the initialization (Salakhutdinov et al., 2003) <span id="cite1a"></span><a href="#ref1a">[1a]</a>. Crucially, even when the per-state emission distribution is itself convex (as the Bernoulli GLM is), marginalizing over the hidden states destroys global convexity (Salakhutdinov et al., 2003). <span id="cite1b"></span><a href="#ref1b">[1b]</a>.

This is also why EM is structured the way it is. The E-step freezes the assignment of trials to states as the soft responsibilities $\gamma_{t,k}$, and given those the emission term of the ECLL is $\sum_{t}\sum_{k}\gamma_{t,k}\log p(y_t \mid \boldsymbol{x}_t, \boldsymbol{w}_k)$, which is a weighted sum of concave GLM log-likelihoods, concave again, with an optimum that can be found from any starting point. Each M-step is therefore a well-behaved convex problem; it is the *sequence* of them that depends on where we began, and the fixed point EM settles on is whichever local optimum lies uphill from the initialization.

The practical consequence is a heuristic: refit the model from several initializations and keep the solution with the highest log-likelihood, which is what we do below. In NeMoS the initialization is controlled by the `seed` argument of {class}`GLMHMM <nemos.glm_hmm.GLMHMM>`, whose defaults already give EM a sensible starting point. Every GLM coefficient is an independent draw from a zero-mean Gaussian with a standard deviation of 0.001. Each state's intercept is then set so that the predicted mean matches the mean of the observations: the logit of the empirical choice rate for a Bernoulli, and in general the link function applied to `y.mean()`. The initial distribution starts uniform, and the transition matrix starts sticky, with a self-transition probability of 0.95 encoding the prior belief that states persist across trials.

```{admonition} Choosing the initialization yourself
:class: tip
{meth}`GLMHMM.setup <nemos.glm_hmm.GLMHMM.setup>` selects how each group of parameters is initialized, either by name or with a callable of your own: `initial_proba_init` (`"uniform"`, `"random"`, `"dirichlet"`, `"kmeans"`), `transition_proba_init` (`"sticky"`, `"uniform"`, `"random"`, `"dirichlet"`, `"kmeans"`), `glm_params_init` (`"random"`, `"kmeans"`) and `scale_init` (`"constant"`, `"kmeans"`), each with a matching `..._kwargs` dictionary for its arguments. The `"kmeans"` options are worth knowing about here: they cluster the data first and start each state from its own cluster, which separates the states from the outset rather than relying on small random noise to pull them apart.

If you would rather skip initialization altogether, pass `init_params` to {meth}`fit <nemos.glm_hmm.GLMHMM.fit>`. It takes the tuple `(coef, intercept, scale, initial_prob, transition_prob)` and bypasses the initializers entirely, which is what you want when you are restarting from an earlier fit or testing whether EM recovers a known solution from a known starting point.
```
 


### Fitting our simulated data
Let us put EM to work on the choices we simulated above. We instantiate a fresh {class}`GLMHMM <nemos.glm_hmm.GLMHMM>`, tell it how many states to look for, and call `fit` — which runs the E- and M-steps to convergence internally — from several seeds, keeping the run with the highest log-likelihood.



```{admonition} Why the number of states has to be chosen, not learned
:class: note
The number of hidden states $K$ is a hyperparameter: it is standardly assumed known before fitting begins (Ibe, 2013) <span id="cite2"></span><a href="#ref2">[2]</a>, and NeMoS requires it as the `n_states` argument. There are two separate reasons for this.

1. The first is structural. $K$ determines the *shape* of everything we are fitting: $\boldsymbol{\pi}$ has $K$ entries, $\boldsymbol{A}$ is $K \times K$, and there is one weight vector per state. EM searches a parameter space of fixed dimension, so that space does not even exist until $K$ is pinned down.

2. The second is statistical: the likelihood cannot arbitrate between values of $K$, because it never decreases as $K$ grows. The reason is that the models are nested. A $K$-state model is a special case of a $(K+1)$-state model: give the extra state zero probability of ever being occupied — so that it is simply never visited — and you recover the smaller model exactly. Because the smaller model is always available to the larger one, the *globally* maximized likelihood can never decrease as $K$ grows, so at the global optimum a likelihood-only comparison would always return the largest $K$ on offer. $K$ therefore has to be judged by a criterion that charges for complexity, not by the one used to fit the parameters.

How much data a given $K$ needs is worth a thought of its own. The parameter count grows quadratically, $(K-1) + K(K-1) + K(M+1)$ for $M$ inputs, but the more useful question is how much data informs each group of them. A state's weights are estimated only from the trials spent in that state. Writing $\rho_k$ for the fraction of trials the chain spends in state $k$, that is about $T\rho_k$ trials, so trials per state fall as states are added. Its transition probabilities are estimated from the number of times the chain *leaves* that state, roughly $T\rho_k(1 - A_{kk})$, which is a much smaller number whenever states are sticky: with the dwell times we chose here, our 3000 trials give about 80 departures per state.

In practice, one fits models across a range of $K$ and compares them with a criterion that penalizes complexity (AIC, BIC) or measures generalization (held-out log-likelihood, cross-validation). These comparisons are worth pairing with a recovery analysis: simulate data from plausible parameters at the number of trials actually available, then check whether fitting recovers those parameters. 
```

Now we can see some code to fit our model to the simulated data and try to recover the initial parameters. 

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

# put the arbitrary state labels into a canonical order to compare with truth
best_model = tutorial_utils.relabel_by_bias(best_model)

fig = tutorial_utils.plot_param_recovery(
    true_coef, true_intercept,
    np.asarray(best_model.coef_), np.asarray(best_model.intercept_),
    true_transition, np.asarray(best_model.transition_prob_),
    state_labels=["engaged", "biased-left", "biased-right"],
)
```

EM has recovered the generative model: one engaged state with a large stimulus weight and near-zero bias, two biased states of opposite sign whose choices barely depend on the stimulus, and a transition matrix with a heavy diagonal, closely matching the one we simulated from.

## References
<a id="ref1a"><a href="#cite1a">[1a]</a> <a id="ref1b"><a href="#cite1b">[1b]</a> Salakhutdinov, R., Roweis, S. T., & Ghahramani, Z. (2003). Optimization with EM and expectation-conjugate-gradient. In Proceedings of the 20th International Conference on Machine Learning (ICML-03) (pp. 672–679).

<a id="ref2"><a href="#cite2">[2]</a> [Ibe, O. C. (2013). Hidden Markov models. In Markov Processes for Stochastic Modeling (2nd ed., Chapter 14, pp. 417–451). Elsevier.](https://doi.org/10.1016/B978-0-12-407795-9.00014-1)
