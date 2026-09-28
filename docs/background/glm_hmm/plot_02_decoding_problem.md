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
# GLM-HMMs: The Decoding Problem

Once a GLM-HMM is fitted, we can ask which strategy the animal was using on each trial. This page answers that question with the two standard readouts, the smoothed posterior and the Viterbi path. It picks up from [the learning problem](plot_01_learning_problem.md): the cells below regenerate the simulated session and refit the model, so the page runs on its own.

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

## Reading out the states
Fitting gives us the parameters, but knowing which strategy the animal used on each trial is a question about the states. We can answer this question with two different NeMoS methods:

{meth}`smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>` is the forward-backward smoother: it returns the **per-trial marginal** posterior $\gamma_{t,k} = p(z_t = k \mid \boldsymbol{y})$, the probability of each state at each trial *considered on its own*, given the whole sequence. It is the right tool when you care about per-trial confidence, including the fact that some trials are genuinely ambiguous (posterior near 0.5). If you take the most probable state at every trial you do get a sequence, but it is stitched together from independent per-trial decisions and need not be a self-consistent path: in principle it can include a transition the model considers impossible (a step from state $j$ to state $k$ with $A_{jk}=0$), because the marginals never look at two trials jointly.

{meth}`decode_state <nemos.glm_hmm.GLMHMM.decode_state>` instead implements the [**Viterbi algorithm**](https://en.wikipedia.org/wiki/Viterbi_algorithm), which returns the single most likely *whole sequence* of states — the joint MAP path $\arg\max_{\boldsymbol{z}} p(\boldsymbol{z} \mid \boldsymbol{y})$. 

Let us compute both on our simulated data and compare them against the true states, which we know but the model never saw. By default `decode_state` returns a one-hot encoding; `state_format="index"` gives the integer state label per trial instead.

```{code-cell} ipython3
# per-trial smoothed posterior over states (forward-backward)
posteriors = np.asarray(best_model.smooth_proba(X, y))

# the state sequence the posterior implies: on each trial, the state holding
# the most probability mass
posterior_states = posteriors.argmax(axis=1)

# single most likely state sequence (Viterbi)
viterbi_states = np.asarray(
    best_model.decode_state(X, y, state_format="index")
).astype(int)

fig = tutorial_utils.plot_state_inference(
    z_true,
    posteriors,
    posterior_states,
    viterbi_states,
    state_labels=["engaged", "biased-left", "biased-right"],
    window=(0, 200),
)
```

These are the same 200 trials shown in [the Introduction](README.md). 

The top panel shows the ground truth and the two readouts, with a short black tick inside each row marking every trial where readout assigns to the wrong state. Note that `smooth_proba` does not return a state sequence; instead, it returns probabilities, as we can see in the bottom panel, and the row shown in the top panel corresponding to `smooth_proba`  is what we get by taking the most likely state on each trial, `posteriors.argmax(axis=1)`. 

The bottom panel plots one posterior curve per state, and the three curves sum to one on every trial. They put almost all of their mass near 0 or 1, and one state dominates at a time. When the states separate this cleanly, the two readouts agree on nearly every trial; they differ at ambiguous switch points.
