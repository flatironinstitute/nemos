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

# GLM-HMMs: An Introduction

This page is the conceptual and mathematical companion to the applied tutorial [Infer behavioral strategies during decision making with GLM-HMMs](../../tutorials/plot_07_behavioral_states.md), which fits a Bernoulli GLM-HMM to a mouse's choice data from the International Brain Laboratory. Here we explain the model and how it is fit. We assume you are comfortable with {ref}`GLMs <nemos_glm>` but not necessarily with hidden Markov models.

Our running motivation is the perceptual decision-making task of the applied tutorial: on each trial a mouse sees a stimulus and makes a binary choice, and it performs hundreds of such trials per session across many sessions. 

```{figure} ../../assets/ibl_task.svg
:width: 640px
:alt: Task illustration
:align: center
IBL task. On each trial, a mouse is shown a grating, a patch of light and dark stripes, whose contrast sets how visible it is against the grey background, on the left or the right of a screen, and reports where it saw it by turning a wheel. A correct report earns a reward, while a wrong report yields a timeout and noise burst (The International Brain Laboratory et al., 2021) <span id="cite1a"></span><a href="#ref1a">[1a]</a>. Redrawn after The International Brain Laboratory et al. (2021) <span id="cite1b"></span><a href="#ref1b">[1b]</a>.
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
GLM-HMMs, also known as input-output HMM models (Bengio & Frasconi, 1995) <span id="cite2"></span><a href="#ref2">[2]</a>, let us describe how hidden latent states shape observable behavioral (Ashwood et al., 2022) <span id="cite3a"></span><a href="#ref3a">[3a]</a> and neural (Escola et al., 2011) <span id="cite4a"></span><a href="#ref4a">[4a]</a> dynamics. This model consists of multiple GLMs, which relate inputs (task variables or observable strategies) with outputs (observed neural or behavioral information) in a sequence determined by a Hidden Markov Model.

+++

### Markov models
Certain events in neuroscience, such as choices, are sequential: what an animal does on one trial is related to what it did just before. In such cases, a model that treats trials as independent is missing something. The simplest way to model a sequence is a first-order Markov chain, in which each observation (such as an action) depends on the whole past only through the immediately preceding one: $p(y_t \mid y_1, \dots, y_{t-1}) = p(y_t \mid y_{t-1})$. With this assumption, the likelihood of an entire sequence can be factorized into an initial term and one transition term per step:

\begin{equation}
p(y_1, \dots, y_T) = p(y_1) \prod_{t=2}^T p(y_t \mid y_{t-1})
\end{equation}

The initial term tells us the probability of starting in one state or another, while the transition term dictates how we move between states. The simplest thing we could do with our mice is to treat the *choice itself* as the state: two states, L (choose left) and R (choose right). Let us see what the graphical model of such chain would look like ({numref}`markov-chain-choices`).

```{figure} ../../assets/markov_chain_choices.svg
:name: markov-chain-choices
:width: 70%
:alt: A two-state Markov chain over left and right choices
:align: center
A first-order Markov chain over choices, with states L (choose left) and R (choose right). Each arrow carries the probability of the next choice given the current one, so the arrows leaving a state sum to one. Here repeating dominates (0.9 against 0.1): having gone left, the animal is nine times more likely to go left again than to switch.
```

Using this graphical model in the form of a transition matrix alongside the initial probabilities, we can sample a sequence of choices.

```{code-cell} ipython3
# choice 1 = left, choice 0 = right
initial_prob = np.array([0.5, 0.5])
transition_prob = np.array([[0.9, 0.1],   # after a right choice: repeat, or switch
                            [0.1, 0.9]])  # after a left choice: switch, or repeat

# we will simulate a session of 100 trials
n_trials = 100

# the first choice is drawn from the initial distribution
choices = np.zeros(n_trials, dtype=int)
choices[0] = rng.choice(2, p=initial_prob)

# then the current choice selects the 
# row of the transition matrix that the next
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

A Markov chain placed like this, directly on the observations, is quite limiting. Everything about such a chain is an observation: the state sequence it passes through is exactly the sequence we observe (Ibe, 2013) <span id="cite5a"></span><a href="#ref5a">[5a]</a>. But two left choices can be entirely different events. One may be a mouse tracking a high-contrast grating on the left; the next may be the same mouse, ignoring the screen, working through a run of leftward turns. 

What we would really like to say is that the choices are symptoms of something we cannot observe directly: his current state. A chain written over the choices cannot instantiate this, and it has been shown in different studies that mice performing decision tasks fluctuate between engaged and disengaged states lasting minutes at a time (Hulsey et al., 2024) <span id="cite6"></span><a href="#ref6">[6]</a> and alternate between an engaged strategy and two biased ones that persist for tens to hundreds of trials (Ashwood et al., 2022) <span id="cite3b"></span><a href="#ref3b">[3b]</a>. In conclusion, a correct choice made in one of these states is not the same event as a correct choice made in another.

### Hidden Markov models

A hidden Markov model (HMM) adds a second variable. At each trial $t$ we posit a latent state $z_t$ which is not observed. Two changes follow (Bishop, 2006) <span id="cite7"></span><a href="#ref7">[7]</a>. First, the first-order Markov assumption is now made about the sequence of states rather than about the observations: $z_t$ depends on the past only through $z_{t-1}$. Second, one observation is attached to the chain at each trial: the choice $y_t$ is emitted by the state $z_t$ active on trial $t$. Conditioning on that state renders $y_t$ independent of all other states and observations. We can see this in the graphical model shown in Panel (b) of {numref}`hmm-graphical-model`.

```{figure} ../../assets/hmm_graphical_model.svg
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

The factorization above is also a recipe for sampling. Draw the first state from $\boldsymbol{\pi}$; draw each state after it from the row of $\boldsymbol{A}$ that the current state selects; and at every trial draw one observation from the emission distribution of whichever state is active.

Let us do this for three strategies: an unbiased strategy, where the animal chooses randomly left or right; a biased-left strategy, in which it mostly turns left; and a biased-right strategy, in which the animal turns right. Since a choice is binary, we can use a Bernoulli as an emission distribution: each state carries its own probability $\phi_k$ of a leftward choice, and the observation on a trial in state $k$ is a single Bernoulli draw with that probability, $y_t \mid z_t = k \sim \mathrm{Bernoulli}(\phi_k)$.

{numref}`hmm-transition-matrix` gives the remaining two objects, $\boldsymbol{\pi}$ and $\boldsymbol{A}$, for these three states.

```{figure} ../../assets/hmm_transition_matrix.svg
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

The top row are the observations, while the bottom row is what we would like to know. Left choices occur under all three states, so no single trial reveals the state: one leftward choice is just as consistent with a state that goes left 85% of the time as with one that goes left half the time. What separates them is the proportion of left choices over a run of consecutive trials, which is why the stretches of a single state in the bottom row become visible only when we look at several trials at once.

Each biased state is captured well by a single number, since turning left most of the time whatever the stimulus does is precisely what being biased means. The unbiased state, however, has been reduced to a coin flip: an animal that follows the task variables closely will also choose left about half the time when the trials are balanced, so no fixed distribution over choices can tell the two apart. The question worth asking is therefore not whether the animal is biased, but in what way occupying one state rather than another changes the relationship between the task variables and the animal's actions. The current model cannot account for the task variables, as those are external inputs that change with time.

Seeing that relationship is precisely what a GLM is for: it expresses behavior as a function of measured inputs, with one weight per input quantifying how strongly that input pushes the animal toward one choice or the other. Making each state's emission a GLM therefore replaces a fixed probability of going left with a rule relating the task variables to the choice; one such rule per state, so that the states differ not in how often the animal goes left, but in what its choices are responding to. The state whose weights make choices follow the stimulus is the one we will come to call engaged.

+++

### GLM as emission probabilities in a GLM-HMM
Of the three objects above, the initial and transition probabilities are fully specified by an HMM, but the emission distribution $p(y_t \mid z_t)$ is left open; that is, it is whatever distribution suits the data being modeled. Emission probabilities describe the probabilities of the possible outcomes from each state. In the case of GLM-HMMs, the emission probabilities are dictated by state-dependent GLMs, which map an input $\mathbf{x}_t$ to an output $y_t$ by rescaling and shifting it, $\mathbf{w}\cdot\mathbf{x}_t + b$, passing the result through a nonlinearity (the inverse link function), and using the result as the parameter of an observation model that generates $y_t$. If you would like a refresher, see the {ref}`conceptual introduction to GLMs <glm_intro_background>` and the {ref}`observation models <observation_models>` NeMoS makes available.

Choosing a GLM here is what makes the model *input-driven*. A plain HMM emission depends only on the state, and all that can be expressed by the state is a fixed distribution over outcomes. A GLM emission depends on the state *and* on that trial's stimulus, so a state expresses a whole input-output rule.. It helps to picture the GLM on its own first:

```{figure} ../../assets/glm_graphical_model.svg
:name: glm-graphical-model
:width: 100%
:alt: Graphical model of a GLM
:align: center
The GLM as a graphical model. One weight vector $\mathbf{w}$ is shared across all trials, and each trial's input $\mathbf{x}_t$ independently produces its observation $y_t$ (shaded, as observed variables are throughout this page). A GLM-HMM keeps this per-trial emission structure, but lets *which* weight vector is used change from trial to trial.
```

A GLM-HMM gives every state one of these GLMs. State $k$ corresponds to a weight vector $\mathbf{w}_k$ and an intercept $b_k$, and whenever the chain sits in state $k$, that state's GLM turns the input $\mathbf{x}_t$ into the choice $y_t$. A $K$-state model therefore holds $K$ separate GLMs, one per strategy, which share a single set of initial and transition probabilities. You can see it in {numref}`glm-hmm-graphical-model`: at each trial the Markov chain picks a state, and that state's GLM converts the stimulus into a choice.

```{figure} ../../assets/glm_hmm_graphical_model.svg
:name: glm-hmm-graphical-model
:width: 100%
:alt: Graphical model GLM-HMM
:align: center
Graphical model of a GLM-HMM. The state $z_t$ at time $t$ depends only on the previous state $z_{t-1}$. Conditioned on $z_t$, the observation $y_t$ is generated by that state's GLM from the input $\mathbf{x}_t$, independently of all previous observations.
```

For a Bernoulli GLM-HMM, NeMoS default, the emission on a single trial is

\begin{align}
y_t \mid \mathbf{x}_t, z_t
\sim \mathrm{Bernoulli}\left(
\sigma(\mathbf{w}_{z_t} \cdot \mathbf{x}_t + b_{z_t})
\right),
\end{align}

where $\mathbf{x}_{t}\in \mathbb{R}^M$ collects the $M$ inputs on trial $t$, the $t$-th row of the design matrix $\mathbf{X}\in \mathbb{R}^{T\times M}$ holding all $T$ trials, and $\mathbf{w}_{z_t}\in \mathbb{R}^M$ and $b_{z_t}\in \mathbb{R}$ are the GLM weights and intercept of the latent state active at that trial. The function $\sigma$ is the inverse link, the nonlinearity that maps the linear term onto the range the observation model needs (here the unit interval, since a Bernoulli needs a probability). For a Bernoulli, NeMoS defaults to the logistic function as nonlinearity, $\sigma(u) = 1/(1 + e^{-u})$, which squashes any real number into $(0, 1)$ and sends $0$ to one half. Writing it out, the probability of $y_t = 1$ (a particular choice in a binary set-up, or a spike count for a time bin) given the input vector $\boldsymbol{x}_t$ and the current state $z_t$ is

\begin{align}
p(y_t=1\mid\boldsymbol{x}_t, z_t = k)  = \frac{1}{1+\exp(-(\boldsymbol{x}_t \cdot \boldsymbol{w}_k + b_k))}.
\end{align}


```{admonition} Other observation models
:class: note
The Bernoulli emission above is the default in NeMoS, but it is only one option. The {ref}`observation models <observation_models>` also include Categorical, Poisson, Gamma, Negative Binomial and Gaussian, each with its own inverse link function, so the same HMM scaffolding can model binary choices, spike counts, continuous signals, and more. Keep in mind that convexity of the per-state emission is not guaranteed for all of them (more on why this matters below). For the neural-activity case, see Escola et al. (2011) <span id="cite4b"></span><a href="#ref4b">[4b]</a>.
```

With state-specific weights and intercepts, the states take on interpretable roles, and the two parameters play different parts. The weights say how strongly the choice follows the inputs, while the intercept sets the side the animal leans to when the inputs are not informative. An engaged state therefore has a large stimulus weight and an intercept near zero, so that its choices track the evidence; a biased state has small weights and a large intercept of one sign, so that its choices lean to that side whatever the stimulus does.

(glm_hmm_example)=
### GLM-HMM example
Let's build a specific Bernoulli GLM-HMM and generate data from it. In this example, the design matrix has a single column ($M = 1$): the signed contrast of the visual stimulus. The sign gives the side the stimulus appeared on (positive for left, negative for right) and the magnitude gives how visible it was (so a trial near zero carries almost no sensory evidence). 

```{code-cell} ipython3
# 3000 trials
T = 3000

# a single stimulus predictor (signed contrast) in [-1, 1]: the sign is the
# side the grating appeared on, the magnitude is how visible it was
rng = np.random.default_rng(2)
X = rng.uniform(-1, 1, size=(T, 1))
```

The left panel below shows the stimulus on a stretch of trials, and the right panel the distribution over the whole experiment.

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax_trace, ax_hist) = plt.subplots(
    1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [2.4, 1]})

trials = np.arange(80)
ax_trace.vlines(trials, 0, X[:80, 0], color="0.6", lw=1.2)
ax_trace.axhline(0, color="k", lw=0.8, ls="--")
ax_trace.set(xlabel="trial", ylabel="signed contrast", xlim=(-1, 80), ylim=(-1.1, 1.1),
             title="the first 80 trials")
ax_trace.text(81, 0.75, "stimulus on the left", fontsize=9, va="center")
ax_trace.text(81, -0.75, "stimulus on the right", fontsize=9, va="center")

ax_hist.hist(X[:, 0], bins=20, color="0.6")
ax_hist.set(xlabel="signed contrast", ylabel="number of trials", yticks=[],
            title=f"all {T} trials")

for ax in (ax_trace, ax_hist):
    sns.despine(ax=ax)
fig.tight_layout()
```

That predictor drives a binary choice $y_t \in \{0, 1\}$, and the animal switches between three strategies: an engaged state with a large stimulus weight and no bias, and two biased states with a weak stimulus weight and a large intercept of either sign.

```{code-cell} ipython3
n_states = 3

# ground-truth parameters. coef_ has shape (n_features, n_states)
# the states are: 0 = engaged, 1 = biased-left, 2 = biased-right

# engaged: strong stimulus weight; biased: weak
true_coef = np.array([[5.0, 0.5, 0.5]])     

# engaged: no bias; biased: strong +/- bias
true_intercept = np.array([0.0, 2.5, -2.5])   

# every state is sticky
true_transition = np.array([[0.94, 0.03, 0.03],
                            [0.05, 0.90, 0.05],
                            [0.05, 0.05, 0.90]])  

# usually start engaged
true_initial = np.array([0.85, 0.10, 0.05])       

# a GLM-HMM whose parameters we set ourselves instead of fitting them
true_model = nmo.glm_hmm.GLMHMM(n_states=n_states, observation_model="Bernoulli")
true_model.coef_ = true_coef
true_model.intercept_ = true_intercept
true_model.transition_prob_ = true_transition
true_model.initial_prob_ = true_initial
# the Bernoulli has no free scale parameter
true_model.scale_ = np.ones(n_states)  

fig = tutorial_utils.plot_state_weights(
    true_coef, true_intercept,
    state_labels=["engaged", "biased-left", "biased-right"],
)
```
The engaged state is composed by a large stimulus weight and a low bias. Thus, observations in this state follow the evidence and favour neither side. The two biased states show a stimulus weight near zero and a large bias of opposite sign. Hence, observations from either of the bias states would ignore the evidence and lean to their own side. We can see the type of behavior this model generates:

```{code-cell} ipython3
# `simulate` walks the chain and emits one choice per trial: z_1 comes from the
# initial distribution, each z_t from the row of the transition matrix that the
# previous state selects, and each choice from the Bernoulli GLM of the state
# active on that trial. It also returns that state's P(choosing left) per trial.
y, prob_left, z_true = true_model.simulate(
    jax.random.PRNGKey(0), X, state_format="index"
)

# Let's plot the weights for our simulation
fig = tutorial_utils.plot_simulated_session(
    X, y, z_true,
    state_labels=["engaged", "biased-left", "biased-right"],
    window=(0, 200),
)
```

Now that we understand what each component of the model does and we have seen a worked example, it is worth naming what we will ultimately want out of the model. The parameters $\boldsymbol{\theta} \equiv \{A,\boldsymbol{\pi},\{\boldsymbol{w}_k, b_k\}_{k=1}^K\}$ are usually a means to an end, as the questions that motivate fitting a GLM-HMM are about the latent states themselves. *When* was each strategy in use? *How much* of the experiment did the animal spend in each one? And *how does behavior differ* between them: is the animal more accurate when engaged, and by how much?

These map onto the two classical HMM problems taken up on the next two pages (Ibe, 2013) <span id="cite5c"></span><a href="#ref5c">[5c]</a>: the **learning problem**, finding the parameters that best explain the observed sequence, which we solve with the EM algorithm; and the **decoding problem**, finding the most likely sequence of hidden states behind the observations, which we solve with the Viterbi algorithm. A third, the **evaluation problem**, which consists of computing how probable the observed sequence is under a given model, is answered by the forward pass along the way, so it needs no page of its own; we pick it up where it appears on [the learning problem](plot_01_learning_problem.md) page.

## Learn More

::::{grid} 1 2 2 2

:::{grid-item-card}

```{toctree}
:maxdepth: 2

plot_01_learning_problem.md
```
:::

:::{grid-item-card}

```{toctree}
:maxdepth: 2

plot_02_decoding_problem.md
```
:::

::::

## Additional resources
- [Bishop (2006) Chapter 13 "Sequential Data"](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf): in particular section 13.2, "Hidden Markov Models", gives a thorough treatment of maximum-likelihood fitting of HMMs, the EM algorithm, the forward-backward algorithm, and the Viterbi algorithm.
- [Ibe (2013) Chapter 14 "Hidden Markov Models"](https://doi.org/10.1016/B978-0-12-407795-9.00014-1): a self-contained treatment of HMMs as a class of Markov process: the underlying assumptions, the three fundamental problems (evaluation, decoding, learning), and the forward, backward, Viterbi and Baum-Welch algorithms that solve them.
- [Zoe Ashwood's SSM tutorial on GLM-HMMs](https://github.com/zashwood/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb): an educational notebook that explains GLM-HMMs and their fitting with both maximum-likelihood and MAP estimation.
- [GLM-HMMs blogpost by Camila Ucheoma](https://anneurai.net/2024/01/26/a-glm-hmm-deep-dive/): a summary of Ashwood et al. (2022) with a concise, intuitive explanation of GLM-HMMs.
- The applied companion to this page, [Infer behavioral strategies during decision making with GLM-HMMs](../../tutorials/plot_07_behavioral_states.md): an end-to-end example that fits a Bernoulli GLM-HMM to real IBL mouse-choice data and interprets the states, transition matrix, and posteriors.

## References
<a id="ref1a"><a href="#cite1a">[1a]</a> <a id="ref1b"><a href="#cite1b">[1b]</a> [Ashwood, Z. C., Roy, N. A., Stone, I. R., Laboratory, I. B., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201–212.](https://doi.org/10.1038/s41593-021-01007-z)

<a id="ref2"><a href="#cite2">[2]</a> Bengio, Y., & Frasconi, P. (1995). An input-output HMM architecture. In G. Tesauro, D. S. Touretzky, & T. K. Leen (Eds.), Advances in neural information processing systems (Vol. 7, pp. 427–434). MIT Press.

<a id="ref3a"><a href="#cite3a">[3a]</a> <a id="ref3b"><a href="#cite3b">[3b]</a> [Escola, S., Fontanini, A., Katz, D., & Paninski, L. (2011). Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural Computation, 23(5), 1071–1132.](https://doi.org/10.1162/NECO_a_00118)

<a id="ref4a"><a href="#cite4a">[4a]</a> <a id="ref4b"><a href="#cite4b">[4b]</a> Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

<a id="ref5a"><a href="#cite5a">[5a]</a> <a id="ref5b"><a href="#cite5b">[5b]</a> <a id="ref5c"><a href="#cite5c">[5c]</a> Salakhutdinov, R., Roweis, S. T., & Ghahramani, Z. (2003). Optimization with EM and expectation-conjugate-gradient. In Proceedings of the 20th International Conference on Machine Learning (ICML-03) (pp. 672–679).

<a id="ref6"><a href="#cite6">[6]</a> [The International Brain Laboratory, Aguillon-Rodriguez, V., Angelaki, D., Bayer, H., Bonacchi, N., Carandini, M., Cazettes, F., Chapuis, G., Churchland, A. K., Dan, Y., Dewitt, E., Faulkner, M., Forrest, H., Haetzel, L., Häusser, M., Hofer, S. B., Hu, F., Khanal, A., Krasniak, C., … Zador, A. M. (2021). Standardized and reproducible measurement of decision-making in mice. eLife, 10, e63711.](https://doi.org/10.7554/eLife.63711)

<a id="ref7"><a href="#cite7">[7]</a> [Ibe, O. C. (2013). Hidden Markov models. In Markov Processes for Stochastic Modeling (2nd ed., Chapter 14, pp. 417–451). Elsevier.](https://doi.org/10.1016/B978-0-12-407795-9.00014-1)
