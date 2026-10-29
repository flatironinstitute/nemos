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
# GLM-HMMs: The Decoding Problem

Once a GLM-HMM is fitted, we can ask which state the model was in at each time step $t$. In this example, one time step is one trial and the states are the animal's strategies. NeMoS offers two readouts, the smoothed posterior and the Viterbi path, and they answer different questions. 

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
import pynapple as nap
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

# NeMoS returns JAX arrays, which pynapple converts to NumPy arrays with a
# warning each time; silence those warnings here.
nap.nap_config.suppress_conversion_warnings = True
```
The cells below simulate a session from the GLM-HMM built in {ref}`the GLM-HMM example <glm_hmm_example>` of GLM-HMMs: An Introduction, the same session fitted in {ref}`Fitting our simulated data <glm_hmm_fitting_simulated_data>` of the learning problem, and refit the model as described there, so this page runs on its own.

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

# `simulate` walks the chain and emits one choice per time step: z_1 comes from
# the initial distribution, each z_t from the row of the transition matrix that
# the previous state selects, and each choice from the Bernoulli GLM of the state
# active at that time step. It also returns that state's P(choosing left) at
# each time step.
y, prob_left, z_true = true_model.simulate(
    jax.random.PRNGKey(0), X, state_format="index"
)
```

```{code-cell} ipython3
:tags: [hide-input]

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
```

## Two readouts
Fitting gives us the parameters; reading out the states is a separate question, and NeMoS answers it with two methods.

{meth}`smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>` returns the smoothed posterior $\gamma_{t,k} = p(z_t = k \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta})$: the probability of state $k$ at time step $t$, given all the observations, past and future. To compute it, the forward-backward algorithm marginalizes over the states at all other time steps: it adds up the probabilities of every state sequence that is in state $k$ at time step $t$, whatever states it takes elsewhere. The neighboring states therefore shape $\gamma_{t,k}$, but $\gamma_{t,k}$ does not record which of them go with state $k$. Taking the most probable state at each time step therefore makes $T$ separate decisions.

{meth}`decode_state <nemos.glm_hmm.GLMHMM.decode_state>` runs the Viterbi algorithm <span id="cite1"></span><a href="#ref1">[1]</a>, which returns the single most probable sequence, $\hat{\boldsymbol{z}} = \arg\max_{\boldsymbol{z}} p(\boldsymbol{z} \mid \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\theta})$. Here the states are chosen jointly: the state chosen at $t$ must fit with the states chosen at $t-1$ and $t+1$ through the transition probabilities. 

A state can be the most probable one at time step $t$ and still not lie on the best path, if moving into it from the previous state and out of it to the next one are both unlikely. The sequence of per-step winners ({meth}`smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>`) is therefore not, in general, the most probable sequence ({meth}`decode_state <nemos.glm_hmm.GLMHMM.decode_state>`), as it can contain very unlikely transitions.

```{admonition} A third readout: the filtered posterior
:class: dropdown
Both readouts above condition on all the observations, past and future. Dropping the future observations gives a different quantity, the filtered posterior $p(z_t = k \mid y_{1:t}, \boldsymbol{x}_{1:t}, \boldsymbol{\theta})$, returned by {meth}`filter_proba <nemos.glm_hmm.GLMHMM.filter_proba>`. It is computed by the forward pass alone, and it is the readout to use when the states must be estimated online, as the data arrive.
```

We compute both readouts on the simulated data and compare them with the true states. By default `decode_state` returns a one-hot encoding; `state_format="index"` gives the integer state label at each time step instead.

```{code-cell} ipython3
# smoothed posterior over states at each time step (forward-backward)
posteriors = best_model.smooth_proba(X, y)

# the most probable state at each time step, chosen separately
posterior_states = posteriors.argmax(axis=1)

# single most probable state sequence (Viterbi)
viterbi_states = best_model.decode_state(X, y, state_format="index")

fig = tutorial_utils.plot_state_inference(
    z_true,
    posteriors,
    posterior_states,
    viterbi_states,
    state_labels=["engaged", "biased-left", "biased-right"],
    window=(0, 200),
)
```

The top panel shows the true states and the two readouts over the first 200 time steps, with a black tick in each row wherever that readout assigns the wrong state. `smooth_proba` returns probabilities, not states: its row is the most probable state at each time step, `posteriors.argmax(axis=1)`, and the bottom panel shows the probabilities themselves, one curve per state, summing to one at every time step. 

## Where the two readouts disagree
To get a sense of when the disagreements happen, we use the time-step index as the time axis and work in two steps with [pynapple](https://pynapple.org/), which represents epochs as an {external:class}`IntervalSet <pynapple.IntervalSet>`. First we collect the epochs of agreement: `threshold(0.5)` keeps only the time steps where `agree` is True (stored as 1) and drops those where it is False (0). Then, `time_support` returns one epoch per run of consecutive kept time steps, starting half a step before the run and ending half a step after it. Finally we subtract those epochs from the whole session with {external:meth}`set_diff <pynapple.IntervalSet.set_diff>`, and what is left is where the readouts disagree.

```{code-cell} ipython3
# the time-step index, used as the time axis
t = np.arange(len(y))
agree = posterior_states == viterbi_states

# the whole session, and the epochs in which the two readouts agree
session = nap.IntervalSet(start=t[0], end=t[-1])
agreement = nap.Tsd(t=t, d=agree).threshold(0.5).time_support

# what is left of the session is where they disagree
disagreement = session.set_diff(agreement)

print(f"{(~agree).sum()} of {len(t)} time steps disagree, "
      f"in {len(disagreement)} stretches")
print(disagreement)
```

We can zoom in on one of the disagreements:

```{code-cell} ipython3
# zoom in on the disagreement at time step 235, together with the
# biased-right run that leads up to it, and show the stimulus and the choice
# at each time step
fig = tutorial_utils.plot_state_inference(
    z_true,
    posteriors,
    posterior_states,
    viterbi_states,
    state_labels=["engaged", "biased-left", "biased-right"],
    window=(195, 246),
    stimulus=X,
    choices=y,
)

A = best_model.transition_prob_
print(f"posterior at time step 235: {np.round(posteriors[235], 2)}")
print(f"P(engaged -> biased-right) = {A[0, 2]:.3f}, "
      f"P(biased-right -> engaged) = {A[2, 0]:.3f}")
```

There is a disagreement in $t=253$. The posterior favors biased-right there, so the per-step readout switches to it, but a path through it must switch from engaged to biased-right and straight back, two transitions the fitted model rates as unlikely (printed above), so Viterbi stays in the engaged state. The bottom panel shows why the whole stretch is difficult: left choices on left stimuli at 224 and 226 favor the engaged state, and from 225 to 234 the stimulus is mostly on the right, where both states predict a right choice.


## References
<a id="ref1"><a href="#cite1">[1]</a> [Viterbi algorithm. (n.d.). In Wikipedia. Retrieved October 29, 2026.](https://en.wikipedia.org/wiki/Viterbi_algorithm)
