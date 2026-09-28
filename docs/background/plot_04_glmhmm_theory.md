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

# GLM-HMMs: an overview

This page is the conceptual and mathematical companion to the applied tutorial [Infer behavioral strategies during decision making with GLM-HMMs](../tutorials/plot_07_behavioral_states.md), which fits a Bernoulli GLM-HMM to a mouse's choice data from the International Brain Laboratory. Here we explain the model and how it is fit. We assume you are comfortable with {ref}`GLMs <nemos_glm>` but not necessarily with hidden Markov models.

Our running motivation is the perceptual decision-making task of the applied tutorial: on each trial a mouse sees a stimulus and makes a binary choice, and it performs hundreds of such trials per session across many sessions. 

```{figure} ../assets/ibl_task.svg
:width: 640px
:alt: Task illustration
:align: center
IBL task. On each trial, a mouse is shown a grating on the left or the right of a screen and reports where it saw it by turning a wheel; a correct report earns a reward, while a wrong report yields a timeout and noise burst (The International Brain Laboratory et al., 2021) <span id="cite6"></span><a href="#ref6">[6]</a>. Redrawn after The International Brain Laboratory et al. (2021) <span id="cite6"></span><a href="#ref6">[6]</a>.
```

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

## What are GLM-HMMs?
GLM-HMMs, also known as input-output HMM models (Bengio & Frasconi, 1995) <span id="cite2"></span><a href="#ref2">[2]</a>, let us describe how hidden latent states shape observable behavioral (Ashwood et al., 2022) <span id="cite1a"></span><a href="#ref1a">[1a]</a> and neural (Escola et al., 2011) <span id="cite3a"></span><a href="#ref3a">[3a]</a> dynamics. This model consists of multiple GLMs, which relate inputs (task variables or observable strategies) with outputs (observed neural or behavioral information) in a sequence determined by a Hidden Markov Model.

+++

### Markov Models
Certain events in neuroscience, such as choices, are sequential: what an animal does on one trial is related to what it did just before. In such cases, a model that treats trials as independent is missing something. The simplest way to model a sequence is a first-order Markov chain, in which each observation (such as an action) depends on the whole past only through the immediately preceding one: $p(y_t \mid y_1, \dots, y_{t-1}) = p(y_t \mid y_{t-1})$. With this assumption, the likelihood of an entire sequence can be factorized into an initial term and one transition term per step:

\begin{equation}
p(y_1, \dots, y_T) = p(y_1) \prod_{t=2}^T p(y_t \mid y_{t-1})
\end{equation}

The initial term tells us the probability of starting in one state or another, while the transition term dictates how we move between states. The simplest thing we could do with our mice is to treat the *choice itself* as the state: two states, L (choose left) and R (choose right). Let us see what the graphical model of such chain would look like ({numref}`markov-chain-choices`).

```{figure} ../assets/markov_chain_choices.svg
:name: markov-chain-choices
:width: 70%
:alt: A two-state Markov chain over left and right choices
:align: center
A first-order Markov chain over choices, with states L (choose left) and R (choose right). Each arrow carries the probability of the next choice given the current one, so the arrows leaving a state sum to one. Here repeating dominates (0.9 against 0.1): having gone left, the animal is nine times more likely to go left again than to switch.
```

Using this graphical model in the form of a transition matrix alongside the initial probabilities, we can sample a sequence of choices.

```{code-cell} ipython3
# choice 1 = left, choice 0 = right, as in the applied tutorial
initial_prob = np.array([0.5, 0.5])
transition_prob = np.array([[0.9, 0.1],   # after a right choice: repeat, or switch
                            [0.1, 0.9]])  # after a left choice: switch, or repeat

# we will simulate a session of 100 trials
n_trials = 100

# the first choice is drawn from the initial distribution
choices = np.zeros(n_trials, dtype=int)
choices[0] = rng.choice(2, p=initial_prob)

# then the current choice selects the row of the transition matrix that the next
# choice is drawn from
for t in range(1, n_trials):
    choices[t] = rng.choice(2, p=transition_prob[choices[t - 1]])
```

```{code-cell} ipython3
:tags: [hide-input]

choice_labels = ["right", "left"]
state_colors = ["#4C82C3", "#E07B39"]  # same colors as the diagram above

trials = np.arange(n_trials)
fig, ax = plt.subplots(figsize=(10, 2.4))
for choice, color in enumerate(state_colors):
    ax.fill_between(trials, -0.45, 1.45, where=choices == choice, step="post",
                    color=color, alpha=0.15, linewidth=0)
ax.step(trials, choices, where="post", color="0.25", linewidth=2)
ax.set(xlabel="trial", xlim=(0, n_trials - 1), ylim=(-0.45, 1.45),
       yticks=[0, 1], yticklabels=choice_labels)
ax.tick_params(left=False)
sns.despine(ax=ax, left=True)
fig.tight_layout()
```

```{admonition} Sticky transitions
:class: note

The sampled sequence comes out in somewhat long runs of the same choice, rather than quick alternation between left and right. This occurs because of the diagonal entry of our transition matrix, i.e. the entry that indicates the probability of staying at the same state.
```

However, a Markov chain placed like this, directly on the observations, is quite limiting. Everything about such a chain is *visible*: the state sequence it passes through is exactly the sequence we observe (Ibe, 2013) <span id="cite7a"></span><a href="#ref7a">[7a]</a>. But two left choices can be entirely different events. One may be a mouse tracking a high-contrast grating on the left; the next may be the same mouse, ignoring the screen, working through a run of leftward turns. 

What we would really like to say is that the choices are symptoms of something we cannot observe directly: his current state. A chain written over the choices has room for this, and it has been shown in different studies that mice performing decision tasks fluctuate between engaged and disengaged states lasting minutes at a time (Hulsey et al., 2024) <span id="cite9"></span><a href="#ref9">[9]</a>, and alternate between an engaged strategy and two biased ones that persist for tens to hundreds of trials (Ashwood et al., 2022) <span id="cite1f"></span><a href="#ref1f">[1f]</a> — a correct choice made in one of these states is not the same event as a correct choice made in another.

### Hidden Markov Models

A **hidden Markov model** (HMM) adds a second variable. At each trial $t$ we posit a *latent* state $z_t$ which is not observed. Two changes follow (Bishop, 2006) <span id="cite4"></span><a href="#ref4">[4]</a>. First, the first-order Markov assumption is now made about the sequence of *states* rather than about the observations: $z_t$ depends on the past only through $z_{t-1}$. Second, one observation is attached to the chain at each trial: the choice $y_t$ is *emitted* by the state $z_t$ active on trial $t$. Conditioning on that state renders $y_t$ independent of all other states and observations. We can see this in the graphical model shown in Panel (**b**) of {numref}`hmm-graphical-model`.

```{figure} ../assets/hmm_graphical_model.svg
:name: hmm-graphical-model
:width: 100%
:alt: A Markov chain over observations, and a hidden Markov model
:align: center
(**a**) A first-order Markov chain placed directly on the observations: each $y_t$ depends only on $y_{t-1}$. (**b**) In a hidden Markov model the Markov property is instead asserted of a latent state $z_t$ (unshaded, because it is never observed), and each $y_t$ is emitted from the state active at that trial. Note that the arrows between the observations are gone: in (**b**) they depend on each other only by way of the states.
```

A HMM with $K$ states is fully defined by the three objects labelled in  Panel (**b**) of {numref}`hmm-graphical-model`:

1. an **initial state distribution** $\boldsymbol{\pi}$, a $K$-element vector giving the probability of each state on the first trial,

\begin{equation}
p(z_1 = i) = \pi_i,\quad \sum_{i=1}^{K} \pi_i = 1;
\end{equation}

2. a **transition matrix** $\boldsymbol{A}$, a $K \times K$ matrix whose entries give the probability of moving from one state to another between consecutive trials,

\begin{equation}
p(z_t=j\mid z_{t-1} = i) = A_{ij};
\end{equation}

3. an **emission distribution** $p(y_t \mid z_t)$, describing how the observation on a trial is generated from the state active on that trial.

Together these are the whole model: an HMM is standardly written as the triple $\lambda = (\boldsymbol{A}, \Phi, \boldsymbol{\pi})$ of transition, emission, and initial probabilities. Putting them together, the joint probability of a sequence of observations and states factorizes into one factor for the initial state, one per transition, and one per emission:

\begin{equation}
p(y_1, \dots, y_T, z_1, \dots, z_T) = p(z_1) \left[\prod_{t=2}^T p(z_t \mid z_{t-1})\right] \prod_{t=1}^T p(y_t \mid z_t)
\end{equation}

```{admonition} The assumptions hiding in that factorization
:class: note
Three assumptions are doing the work above (Ibe, 2013) <span id="cite7d"></span><a href="#ref7d">[7d]</a>. The **Markov assumption** — the next state depends only on the current one — is what reduces the transition term to a single factor per step; models where the next state depends on the previous $k$ states are possible but much harder to analyze. The **stationarity assumption** — transition probabilities do not depend on *when* the transition happens — is what lets one fixed matrix $\boldsymbol{A}$ serve for the whole experiment. And the **observation independence assumption** — each observation is independent of the others given its own state — is what turns the emissions into a simple product $\prod_t p(y_t \mid z_t)$. Any structure in the observations must therefore be carried by the states, or else be built into the emission's own inputs.
```

The factorization above is also a recipe for sampling. Draw the first state from $\boldsymbol{\pi}$; draw each state after it from the row of $\boldsymbol{A}$ that the current state selects; and at every trial draw one observation from the emission distribution of whichever state is active.

Let us do this for three strategies: an **unbiased** strategy, where the animal chooses randomly left or right; a **biased-left** strategy, in which it mostly turns left; and a **biased-right** strategy, in which the animal turns right. Since a choice is binary, we can use a **Bernoulli** as an emission distribution: each state carries its own probability $\phi_k$ of a leftward choice, and the observation on a trial in state $k$ is a single Bernoulli draw with that probability, $y_t \mid z_t = k \sim \mathrm{Bernoulli}(\phi_k)$.

{numref}`hmm-transition-matrix` gives the remaining two objects, $\boldsymbol{\pi}$ and $\boldsymbol{A}$, for these three states.

```{figure} ../assets/hmm_transition_matrix.svg
:name: hmm-transition-matrix
:width: 100%
:alt: HMM initial and transition probabilities for three states
:align: center
The initial-state vector, transition matrix of a three-state HMM, graphical mdoel of the three-state HMM transition matrix. Each row of the matrix collects the probabilities of leaving one state and must sum to one; the diagonal entries are the self-transitions (the loops in the diagram), and values near one make each state sticky.
```

We now have everything the factorization asks for, so we can sample a session.

```{code-cell} ipython3
# hidden state 0 = unbiased, 1 = biased-left, 2 = biased-right
# observation 1 = left, 0 = right
strategy_initial_prob = np.array([0.85, 0.10, 0.05])
strategy_transition_prob = np.array([[0.94, 0.03, 0.03],
                                     [0.05, 0.90, 0.05],
                                     [0.05, 0.05, 0.90]])

# Bernoulli emission parameters: P(choosing left | state). The stimulus appears on
# either side equally often, so an unbiased animal goes left about half the time,
# while the biased ones favour their own side whatever the stimulus does.
emission_prob = np.array([0.5, 0.85, 0.15])

# first the latent chain: one draw per trial, each looking back only one trial
n_trials = 200
hidden_states = np.zeros(n_trials, dtype=int)
hidden_states[0] = rng.choice(3, p=strategy_initial_prob)
for t in range(1, n_trials):
    hidden_states[t] = rng.choice(3, p=strategy_transition_prob[hidden_states[t - 1]])

# then the emissions: on each trial, one Bernoulli draw with that state's probability
choices = rng.binomial(1, emission_prob[hidden_states])

strategy_labels = ["unbiased", "biased-left", "biased-right"]
for state, label in enumerate(strategy_labels):
    in_state = hidden_states == state
    print("%-13s %3d trials, chose left on %2.0f%% of them"
          % (label, in_state.sum(), 100 * choices[in_state].mean()))
```

```{code-cell} ipython3
:tags: [hide-input]

strategy_colors = ["#E07B39", "#4F9E4F", "#4C82C3"]  # as in the figure above

trials = np.arange(n_trials)
fig, (ax_obs, ax_state) = plt.subplots(
    2, 1, figsize=(10, 4.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.4]})

ax_obs.step(trials, choices, where="post", color="0.25", linewidth=2)
ax_obs.set(yticks=[0, 1], yticklabels=["right", "left"], ylim=(-0.4, 1.4),
           title="what we observe")

for state, color in enumerate(strategy_colors):
    ax_state.fill_between(trials, state - 0.4, state + 0.4,
                          where=hidden_states == state, step="post",
                          color=color, linewidth=0)
ax_state.set(yticks=[0, 1, 2], yticklabels=strategy_labels, ylim=(-0.6, 2.6),
             xlabel="trial", xlim=(0, n_trials - 1), title="the hidden strategy")

for ax in (ax_obs, ax_state):
    ax.tick_params(left=False)
    sns.despine(ax=ax, left=True)
fig.tight_layout()
```

The top row is everything we would ever get to see; the bottom row is what we would like to know. Left choices occur under all three strategies, so no single trial reveals the state, and only the statistics of a stretch of trials make the blocks visible at all.

Each biased state is captured well by a single number, since turning left most of the time whatever the stimulus does is precisely what being biased means. The unbiased state, however, has been reduced to a coin flip: an animal that follows the task variables closely will also choose left about half the time when the trials are balanced, so no fixed distribution over choices can tell the two apart. The question worth asking is therefore not whether the animal is biased, but in what way occupying one state rather than another changes the relationship between the task variables and the animal's actions. The current model cannot account for the task variables, as those are external inputs that change with time.

Seeing that relationship is precisely what a GLM is for: it expresses behavior as a function of measured inputs, with one weight per input quantifying how strongly that input pushes the animal toward one choice or the other. Making each state's emission a GLM therefore replaces a fixed probability of going left with a rule relating the task variables to the choice — one such rule per state, so that the states differ not in how often the animal goes left, but in what its choices are responding to. The state whose weights make choices follow the stimulus is the one we will come to call engaged.


— a question about a mapping, which a single probability cannot answer. Once the emission becomes a function of that trial's inputs, the model can express such a mapping, and the state in which choices track the task variables is the one we will come to call engaged.


+++

### GLM as emission probabilities in a GLM-HMM
Of the three objects above, the initial and transition probabilities are fully specified by an HMM, but the emission distribution $p(y_t \mid z_t)$ is left open — it is whatever distribution suits the data being modeled. Emission probabilities describe the probabilities of the possible outcomes from each state. In the case of GLM-HMMs, the emission probabilities are dictated by state-dependent GLMs, which map an input $\mathbf{x}_t$ to an output $y_t$ by forming a linear combination $\mathbf{w}\cdot\mathbf{x}_t$, passing it through a nonlinearity (the inverse link function), and using the result as the parameter of an observation model that generates $y_t$. If you would like a refresher, see the {ref}`conceptual introduction to GLMs <glm_intro_background>` and the {ref}`observation models <observation_models>` NeMoS makes available.

Choosing a GLM here is what makes the model *input-driven*. A plain HMM emission depends only on the state, so all a state can express is a fixed distribution over outcomes. A GLM emission depends on the state *and* on that trial's stimulus, so a state expresses a whole input-output rule.. It helps to picture the GLM on its own first:

```{figure} ../assets/glm_graphical_model.svg
:width: 100%
:alt: Graphical model of a GLM
:align: center
The GLM as a graphical model. One weight vector $\mathbf{w}$ is shared across all trials, and each trial's input $\mathbf{x}_t$ independently produces its observation $y_t$ (shaded, as observed variables are throughout this page). A GLM-HMM keeps this per-trial emission structure, but lets *which* weight vector is used change from trial to trial.
```

A **GLM-HMM** makes each state's emission one of these: state $k$ carries its own weight vector $\mathbf{w}_k$, and on a trial where $z_t = k$ the choice is generated by that state's GLM from the input $\mathbf{x}_t$. A $K$-state GLM-HMM therefore contains $K$ independent GLMs — one per strategy — sharing the same HMM scaffolding of initial and transition probabilities. This is the model drawn below: the Markov chain over states selects, at each trial, which GLM converts the stimulus into a choice.

```{figure} ../assets/glm_hmm_graphical_model.svg
:width: 100%
:alt: Graphical model GLM-HMM
:align: center
Graphical model of a GLM-HMM. The state $z_t$ at time $t$ depends only on the previous state $z_{t-1}$. Conditioned on $z_t$, the observation $y_t$ is generated by that state's GLM from the input $\mathbf{x}_t$, independently of all previous observations.
```

For a Bernoulli GLM-HMM — the default in NeMoS, and the setting of the applied tutorial — the emission on a single trial is

\begin{align}
y_t \mid \mathbf{x}_t, z_t
\sim \mathrm{Bernoulli}\left(
\sigma(\mathbf{w}_{z_t} \mathbf{x}_t)
\right),
\end{align}

where $\mathbf{x}_{t}\in \mathbb{R}^M$ is the input at trial $t$ and $\mathbf{w}_{z_t}\in \mathbb{R}^M$ are the GLM weights of the latent state active at that trial. Writing out the logistic inverse link, the probability of $y_t = 1$ (a particular choice in a binary set-up, or a spike count for a time bin) given the input vector $\boldsymbol{x}_t$ and the current state $z_t$ is

\begin{align}
p(y_t=1\mid\boldsymbol{x}_t, z_t = k)  = \frac{1}{1+\exp(-\boldsymbol{x}_t \cdot \boldsymbol{w}_k)}.
\end{align}

With state-specific weights, the states take on interpretable roles. An **engaged** state has a GLM that weights the stimulus heavily, so its choices track the evidence; **biased** states have weights that largely ignore the stimulus in favor of a fixed side.

This is also where it is worth naming what we will ultimately want out of the model. The parameters $\{\boldsymbol{A}, \boldsymbol{\pi}, \{\mathbf{w}_k\}\}$ are usually a means to an end; the questions that motivate fitting a GLM-HMM are about the latent states themselves. *When* was each strategy in use? *How much* of the experiment did the animal spend in each one? And *how does behavior differ* between them — is the animal more accurate when engaged, and by how much?

These map onto the two classical HMM problems that occupy the rest of this page (Ibe, 2013) <span id="cite7f"></span><a href="#ref7f">[7f]</a>: the **learning problem**, finding the parameters that best explain the observed sequence, which we solve below with the EM algorithm; and the **decoding problem**, finding the most likely sequence of hidden states behind the observations, which we solve with the Viterbi algorithm. A third, the **evaluation problem** — computing how probable the observed sequence is under a given model — turns out to be a sub-step of both, and is what the forward pass computes along the way.

```{admonition} Other observation models
:class: note
The Bernoulli emission above is the default in NeMoS, but it is only one option. The {ref}`observation models <observation_models>` also include Categorical, Poisson, Gamma, Negative Binomial and Gaussian, each with its own inverse link function, so the same HMM scaffolding can model binary choices, spike counts, continuous signals, and more. Keep in mind that convexity of the per-state emission is not guaranteed for all of them (more on why this matters below). For the neural-activity case, see Escola et al. (2011) <span id="cite3b"></span><a href="#ref3b">[3b]</a>.
```

+++

### A concrete generative example
To make the three objects tangible, let us build a specific Bernoulli GLM-HMM and generate data from it: a single stimulus predictor $x_t \in [-1, 1]$ (signed sensory evidence) drives a binary choice $y_t \in \{0, 1\}$, and the animal switches between three strategies — an **engaged** state with a large stimulus weight and no bias, and two **biased** states with a weak stimulus weight and a large intercept of either sign. These mirror the strategies Ashwood et al. (2022) <span id="cite1b"></span><a href="#ref1b">[1b]</a> found in mice, and knowing the exact generative parameters is what will let us check parameter recovery and state decoding later.

```{code-cell} ipython3
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

y, z_true = tutorial_utils.simulate_glm_hmm(
    X, true_coef, true_intercept, true_transition, true_initial, seed=0
)

fig = tutorial_utils.plot_simulated_session(
    X, y, z_true,
    state_labels=["engaged", "biased-left", "biased-right"],
    window=(0, 200),
)
```

Two features of the simulated data are worth pausing on, because they are what the machinery below has to exploit. Because we chose sticky transitions, the state sequence stays in one strategy for many trials at a time rather than flickering — the blocks in the top panel. And within those blocks the *relationship* between stimulus and choice changes: in the engaged state the choices follow the sign of the stimulus, while in the biased states they cluster on one side almost regardless of it. Nothing in the observed data labels these blocks for us, which is precisely the inference problem.

```{admonition} Handling session breaks
:class: note
Real experiments come in separate sessions recorded on different days, and we do not want the model to treat the last trial of one session and the first trial of the next as a genuine transition. NeMoS handles this through the `session_starts` argument to `fit`, `smooth_proba`, `decode_state`, and `simulate`: the **transition** term is dropped across session boundaries (so the state chain is reset at each session start), while the **emission** term still applies to every trial. Concretely, `session_starts` accepts a boolean/0-1 array of shape `(n_samples,)`, an integer array of session-start indices, or — when `X` or `y` is a pynapple `Tsd`/`TsdFrame` — a `pynapple.IntervalSet`.

This is also why NeMoS is strict about missing values: the forward-backward pass cannot step over a `NaN` in the interior of a chain, so NaNs are only allowed at a chain's *border*. A lagged predictor that is undefined at each session's first trial therefore has to be handled either with pynapple per-session epochs (convolution is epoch-aware and NaN-pads each epoch's start) or by dropping the first trial of each session. Our example here is a single continuous session, so we omit `session_starts`; the applied tutorial shows the multi-session case in full.
```

## How to fit a GLM-HMM?

When fitting a GLM-HMM, the goal is to learn the set of parameters $\boldsymbol{\theta} \equiv \{A,\boldsymbol{\pi},\{\boldsymbol{w}_k\}_{k=1}^K\}$ — the transition matrix $\boldsymbol{A} \in \mathbb{R}^{K \times K}$, the initial state distribution $\boldsymbol{\pi} \in \mathbb{R}^K$, and the per-state weights $\{\boldsymbol{w}_k\}_{k=1}^K$ with $\boldsymbol{w}_k \in \mathbb{R}^M$ — that maximize the likelihood of the observed data. Because the states are hidden, the likelihood of the observations alone is obtained by marginalizing the joint distribution over every possible state sequence:

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

```{admonition} The smoothed posterior is what smooth_proba returns
:class: tip
The per-trial smoothed posterior $\gamma_{t,k}$ is not just an internal EM quantity — it is directly useful for interpretation, and NeMoS exposes it. {meth}`GLMHMM.smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>` *is* the forward-backward smoother: it runs both passes at the fitted parameters and returns the array of per-trial state posteriors $\gamma_{t,k}$, of shape `(n_samples, n_states)`, each row summing to one. We use it below to read out when each strategy was active.
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
The closed-form updates above are maximum-likelihood estimates, which can be poorly behaved when a state or a transition is rarely visited — an entry can be driven all the way to 0 or 1. A **Dirichlet prior** on the initial and transition probabilities regularizes this. The Dirichlet is the natural prior over a categorical distribution (each row of $\boldsymbol{A}$, and $\boldsymbol{\pi}$, is one such distribution): its concentration parameters act like pseudo-counts, keeping every entry a valid probability and pulling the estimates away from degenerate 0-or-1 rows and, if desired, toward stickiness. In NeMoS this turns the M-step updates into *maximum a posteriori* updates, and you control it through the `dirichlet_initial_proba` and `dirichlet_transition_proba` arguments of {class}`GLMHMM <nemos.glm_hmm.GLMHMM>` (each a vector/matrix of Dirichlet concentration parameters; left `None`, a flat, uninformative prior is used). The GLM weights, in turn, are regularized through the `regularizer` argument, just as for a plain GLM.
```

### A caveat: the likelihood is non-convex
EM is guaranteed to increase (or hold) the marginal log-likelihood at every iteration, but that does *not* mean it finds the best possible solution. The GLM-HMM marginal likelihood is **non-convex**: it has multiple local optima, and EM converges to whichever one lies uphill from where it started, so the result depends on the initialization (Salakhutdinov et al., 2003) <span id="cite5a"></span><a href="#ref5a">[5a]</a>. Crucially, even when the per-state emission distribution is itself convex (as the Bernoulli GLM is), marginalizing over the hidden states destroys global convexity, so a convex emission does not rescue us (Salakhutdinov et al., 2003) <span id="cite5b"></span><a href="#ref5b">[5b]</a>.

The practical consequence is a workflow rather than a fix: **refit the model from several initializations and keep the solution with the highest log-likelihood**, which is what we do below. In NeMoS the initialization is controlled by the `seed` argument of {class}`GLMHMM <nemos.glm_hmm.GLMHMM>`, whose defaults already give EM a sensible starting point — small random GLM coefficients with each state's intercept at the empirical choice rate, a uniform initial distribution, and a sticky transition matrix (0.95 self-transition probability, encoding the prior belief that states persist across trials).

### Fitting our simulated data
Let us put EM to work on the choices we simulated above. We instantiate a fresh {class}`GLMHMM <nemos.glm_hmm.GLMHMM>`, tell it how many states to look for, and call `fit` — which runs the E- and M-steps to convergence internally — from several seeds, keeping the run with the highest log-likelihood.

One more wrinkle before we can compare against the ground truth: the state labels are arbitrary. EM has no way of knowing we called state 0 "engaged", so two runs can find the same solution with the columns permuted. We therefore put the recovered states into a canonical order — engaged (smallest bias) first, then the biased states by decreasing bias — before comparing. Because `relabel_by_bias` reorders the fitted attributes themselves, everything the model produces afterwards, including the posteriors and decoded states below, follows the same ordering.

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


```{admonition} Why the number of states has to be chosen, not learned
:class: note
The number of hidden states $K$ is a hyperparameter: it is standardly assumed known before fitting begins (Ibe, 2013) <span id="cite7e"></span><a href="#ref7e">[7e]</a>, and NeMoS requires it as the `n_states` argument. There are two separate reasons for this.

The first is structural. $K$ determines the *shape* of everything we are fitting: $\boldsymbol{\pi}$ has $K$ entries, $\boldsymbol{A}$ is $K \times K$, and there is one weight vector per state. EM searches a parameter space of fixed dimension, so that space does not even exist until $K$ is pinned down.

The second is why we cannot simply add $K$ to the search. The likelihood we maximize when fitting is computed on the same data used to fit, and such an in-sample score is biased in favor of more complex models: a model fits the sample it was trained on better than it fits a fresh one, and that optimism grows with the number of free parameters (Pohle et al., 2017) <span id="cite8a"></span><a href="#ref8a">[8a]</a>. Adding states therefore tends to buy apparent improvement whether or not the extra states are real.

The reason more states can always at least tie is that the models are nested. A $K$-state model is a special case of a $(K+1)$-state model: give the extra state zero probability of ever being occupied — so that it is simply never visited — and you recover the smaller model exactly. Because the smaller model is always available to the larger one, the *globally* maximized likelihood can never decrease as $K$ grows, so at the global optimum a likelihood-only comparison would always return the largest $K$ on offer. $K$ therefore has to be judged by a criterion that charges for complexity, not by the one used to fit the parameters.

(In practice EM finds a local, not global, optimum — see the non-convexity section below — so an actual $(K+1)$-state fit can come out *worse* than a $K$-state one. That is a fact about the optimizer, not a reason to trust the raw likelihood for choosing $K$.)

Note that this is a statistical problem, not a computational one — forward-backward and Viterbi cost $\mathcal{O}(K^2 T)$ for $K$ states and $T$ trials, so more states are more expensive but never intractable.

In practice, then, one fits models across a range of $K$ and compares them with a criterion that penalizes complexity (AIC, BIC) or measures generalization (held-out log-likelihood, cross-validation). Be aware that these criteria tend to favor more states than are scientifically interpretable, especially when the model is misspecified in other ways, so they are best combined with a judgement about whether the extra states mean anything (Pohle et al., 2017) <span id="cite8b"></span><a href="#ref8b">[8b]</a>. There are also Bayesian nonparametric variants — the hierarchical Dirichlet process HMM, or "infinite HMM" — that treat the number of states as itself unknown and infer it, at the cost of a different inference machinery than the EM used here.
```

## Reading out the states
Fitting gives us the parameters, but the questions we posed earlier were about the states. NeMoS answers them with two complementary tools, both built on the machinery above, and the difference between them matters.

{meth}`smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>` is the forward-backward smoother: it returns the **per-trial marginal** posterior $\gamma_{t,k} = p(z_t = k \mid \boldsymbol{y})$, the probability of each state at each trial *considered on its own*, given the whole sequence. It is the right tool when you care about per-trial confidence, including the fact that some trials are genuinely ambiguous (posterior near 0.5). If you take the most probable state at every trial you do get a sequence, but it is stitched together from independent per-trial decisions and need not be a self-consistent path: in principle it can include a transition the model considers impossible (a step from state $j$ to state $k$ with $A_{jk}=0$), because the marginals never look at two trials jointly.

{meth}`decode_state <nemos.glm_hmm.GLMHMM.decode_state>` instead implements the **Viterbi algorithm**, which returns the single most likely *whole sequence* of states — the joint MAP path $\arg\max_{\boldsymbol{z}} p(\boldsymbol{z} \mid \boldsymbol{y})$ — and is therefore guaranteed to be a legal, globally coherent trajectory. That makes it the right tool when you want to segment the experiment into contiguous blocks of one strategy and compute per-block statistics such as accuracy or occupancy. Structurally, Viterbi is almost identical to the forward pass, with one change: replace the *sum* over previous states with a *max*, and keep a back-pointer to the previous state that achieved it. Defining

\begin{equation}
\delta_{t,k} = \max_{z_1,\dots,z_{t-1}} p(\boldsymbol{y}_{1:t}, z_1,\dots,z_{t-1}, z_t = k),
\end{equation}

the recursion is

\begin{equation}
\delta_{t,k} = p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}_k)\ \max_{j}\left[\delta_{t-1,j}\,A_{jk}\right],
\end{equation}

storing at each step the $j$ that attained the maximum. A forward sweep fills in $\delta$; a backward sweep follows the back-pointers from the best final state to reconstruct the optimal path. This "max-product" (as opposed to "sum-product") dynamic program is why Viterbi returns one coherent sequence rather than a bag of per-trial marginals.

Let us compute both on our simulated data and compare them against the true states, which we know but the model never saw. By default `decode_state` returns a one-hot encoding; `state_format="index"` gives the integer state label per trial instead.

```{code-cell} ipython3
# per-trial smoothed posterior over states (forward-backward)
posteriors = np.asarray(best_model.smooth_proba(X, y))

# single most likely state sequence (Viterbi)
viterbi_states = np.asarray(
    best_model.decode_state(X, y, state_format="index")
).astype(int)

fig = tutorial_utils.plot_state_inference(
    z_true,
    posteriors,
    viterbi_states,
    state_labels=["engaged", "biased-left", "biased-right"],
    window=(0, 200),
)
```

These are the same 200 trials plotted at the start, now seen through the fitted model rather than the generative one. Both decoders recover the hidden strategy on the large majority of trials. The posteriors spend most of their mass near 0 or 1, with one state's curve dominating at a time, and the Viterbi path snaps between strategies at roughly the trials where the true state switches — the confident, multi-trial state assignments we would hope to see, and the same pattern we observe on real data in the applied tutorial. Where states are this well separated the two readouts mostly coincide; they diverge mainly around ambiguous switch points, where insisting on a coherent path and taking the per-trial best state can disagree.

## Additional resources
- [Bishop (2006) Chapter 13 "Sequential Data"](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf): in particular section 13.2, "Hidden Markov Models", gives a thorough treatment of maximum-likelihood fitting of HMMs, the EM algorithm, the forward-backward algorithm, and the Viterbi algorithm.
- [Ibe (2013) Chapter 14 "Hidden Markov Models"](https://doi.org/10.1016/B978-0-12-407795-9.00014-1): a self-contained treatment of HMMs as a class of Markov process — the underlying assumptions, the three fundamental problems (evaluation, decoding, learning), and the forward, backward, Viterbi and Baum-Welch algorithms that solve them.
- [Zoe Ashwood's SSM tutorial on GLM-HMMs](https://github.com/zashwood/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb): an educational notebook that explains GLM-HMMs and their fitting with both maximum-likelihood and MAP estimation.
- [GLM-HMMs blogpost by Camila Ucheoma](https://anneurai.net/2024/01/26/a-glm-hmm-deep-dive/): a summary of Ashwood et al. (2022) with a concise, intuitive explanation of GLM-HMMs.
- The applied companion to this page, [Infer behavioral strategies during decision making with GLM-HMMs](../tutorials/plot_07_behavioral_states.md): an end-to-end example that fits a Bernoulli GLM-HMM to real IBL mouse-choice data and interprets the states, transition matrix, and posteriors.

## References
<a id="ref1a"><a href="#cite1a">[1a]</a> <a id="ref1b"><a href="#cite1b">[1b]</a> <a id="ref1c"><a href="#cite1c">[1c]</a> <a id="ref1d"><a href="#cite1d">[1d]</a> <a id="ref1e"><a href="#cite1e">[1e]</a> <a id="ref1f"><a href="#cite1f">[1f]</a> [Ashwood, Z. C., Roy, N. A., Stone, I. R., Laboratory, I. B., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201–212.](https://doi.org/10.1038/s41593-021-01007-z)

<a id="ref2"><a href="#cite2">[2]</a> Bengio, Y., & Frasconi, P. (1995). An input-output HMM architecture. In G. Tesauro, D. S. Touretzky, & T. K. Leen (Eds.), Advances in neural information processing systems (Vol. 7, pp. 427–434). MIT Press.

<a id="ref3a"><a href="#cite3a">[3a]</a> <a id="ref3b"><a href="#cite3b">[3b]</a> [Escola, S., Fontanini, A., Katz, D., & Paninski, L. (2011). Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural Computation, 23(5), 1071–1132.](https://doi.org/10.1162/NECO_a_00118)

<a id="ref4"><a href="#cite4">[4]</a> Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

<a id="ref5a"><a href="#cite5a">[5a]</a> <a id="ref5b"><a href="#cite5b">[5b]</a> Salakhutdinov, R., Roweis, S. T., & Ghahramani, Z. (2003). Optimization with EM and expectation-conjugate-gradient. In Proceedings of the 20th International Conference on Machine Learning (ICML-03) (pp. 672–679).

<a id="ref6"><a href="#cite6">[6]</a> [The International Brain Laboratory, Aguillon-Rodriguez, V., Angelaki, D., Bayer, H., Bonacchi, N., Carandini, M., Cazettes, F., Chapuis, G., Churchland, A. K., Dan, Y., Dewitt, E., Faulkner, M., Forrest, H., Haetzel, L., Häusser, M., Hofer, S. B., Hu, F., Khanal, A., Krasniak, C., … Zador, A. M. (2021). Standardized and reproducible measurement of decision-making in mice. eLife, 10, e63711.](https://doi.org/10.7554/eLife.63711)

<a id="ref7a"><a href="#cite7a">[7a]</a> <a id="ref7b"><a href="#cite7b">[7b]</a> <a id="ref7c"><a href="#cite7c">[7c]</a> <a id="ref7d"><a href="#cite7d">[7d]</a> <a id="ref7e"><a href="#cite7e">[7e]</a> <a id="ref7f"><a href="#cite7f">[7f]</a> [Ibe, O. C. (2013). Hidden Markov models. In Markov Processes for Stochastic Modeling (2nd ed., Chapter 14, pp. 417–451). Elsevier.](https://doi.org/10.1016/B978-0-12-407795-9.00014-1)

<a id="ref8a"><a href="#cite8a">[8a]</a> <a id="ref8b"><a href="#cite8b">[8b]</a> [Pohle, J., Langrock, R., van Beest, F. M., & Schmidt, N. M. (2017). Selecting the number of states in hidden Markov models: Pragmatic solutions illustrated using animal movement. Journal of Agricultural, Biological and Environmental Statistics, 22(3), 270–293.](https://doi.org/10.1007/s13253-017-0283-8)

<a id="ref9"><a href="#cite9">[9]</a> [Hulsey, D., Zumwalt, K., Mazzucato, L., McCormick, D. A., & Jaramillo, S. (2024). Decision-making dynamics are predicted by arousal and uninstructed movements. Cell Reports, 43(2), 113709.](https://doi.org/10.1016/j.celrep.2024.113709)
