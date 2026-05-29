# Making `ClassifierGLMHMM` functional

A design/implementation plan for extending `GLMHMM` to categorical (multi-class)
observations, mirroring the way `ClassifierGLM` extends `GLM`.

Status: `classifier_glm_hmm.py` currently has only the public-API skeleton
(`__init__`, `fit`, `score`). The plan below lists what is missing, what already
works for free, and — importantly — the two places where the core EM machinery
makes an assumption that is *wrong* for the categorical case and must be changed.

---

## 1. Mental model: where the extra axis goes

`ClassifierGLM` adds an `n_classes` axis to the GLM (`coef: (n_features, n_classes)`)
and uses `log_softmax` + one-hot `y`. `GLMHMM` adds an `n_states` axis
(`coef: (n_features, n_states)`) and keeps states as the **last** axis everywhere.

`ClassifierGLMHMM` needs **both** axes. The existing
`ClassifierGLMHMMValidator` (in [validation.py](validation.py#L193-L233)) already
commits to the layout:

| tensor              | `GLMHMM` (single)         | `ClassifierGLMHMM`                       |
| ------------------- | ------------------------- | ---------------------------------------- |
| `coef`              | `(n_features, n_states)`  | `(n_features, n_classes, n_states)`      |
| `intercept`         | `(n_states,)`             | `(n_classes, n_states)`                  |
| `y` (one-hot)       | `(n_time,)`               | `(n_time, n_classes)`                    |
| rate per state      | `(n_time, n_states)`      | `(n_time, n_classes, n_states)`          |
| posteriors          | `(n_time, n_states)`      | `(n_time, n_states)`  *(unchanged)*      |

**Key consequence:** in this layout `n_states` is the last axis and `n_classes`
is the *second-to-last* axis. The categorical machinery in `nemos` (e.g.
`CategoricalObservations._negative_log_likelihood`, `log_softmax`) was written
assuming **classes are the last axis** (see
[observation_models.py:1813](../observation_models.py#L1813),
`jnp.sum(y * predicted_rate, axis=-1)`). That mismatch is the source of the two
required core changes in §4.

The convenient part: because the rate gains an extra (class) axis, its shape
`(n_time, n_classes, n_states)` looks exactly like the **population** rate shape
`(n_time, n_neurons, n_states)`. So `is_population_glm = y.ndim > 1` is `True`
for categorical, and most population code paths (einsum in `compute_rate_per_state`,
state-axis vmapping, posterior weighting, initialization) just work with
`n_classes` playing the role of `n_neurons`.

---

## 2. What already works for free

These need no changes — they key off shapes or the `is_population_glm` flag and
behave correctly with `n_classes` in the neuron slot:

- **`compute_rate_per_state`** ([utils.py:50-60](utils.py#L50-L60)) — `coef.ndim > 2`
  triggers the einsum `"ik,kjw->ijw"`, producing `(n_time, n_classes, n_states)`. ✅
- **M-step coefficient objective** — `prepare_mstep_nll_objective_param`
  ([algorithm_configs.py:283-346](algorithm_configs.py#L283-L346)) vmaps the
  categorical NLL over the state axis (`state_axes = 2`), each slice is
  `(n_time, n_classes)`, the categorical NLL sums over `axis=-1` (classes) →
  `(n_time, n_states)`, then `_posterior_weighted_objective_impl`
  ([algorithm_configs.py:78-84](algorithm_configs.py#L78-L84)) uses a robust
  `if nll.ndim > 2` guard before multiplying by posteriors. ✅ (No spurious
  neuron-sum because `nll.ndim == 2` here.)
- **Parameter initialization** — `random_glm_params_init` sets
  `n_neurons = 1 if is_one_dim else y.shape[1]`, so one-hot `y` yields
  `n_neurons = n_classes` and `coef: (n_features, n_classes, n_states)`. ✅
  (Caveat: the intercept "mean-rate" init should be sanity-checked under
  `log_softmax`; see §5.)
- **Scale handling** — `CategoricalObservations._separable_scale = True` and
  `estimate_scale` returns ones; `has_fixed_scale` is `False` but the analytical
  update dict has no entry, so the M-step would *numerically* "optimize" a scale
  that has no effect. See §4.3 — better to make scale genuinely fixed.

---

## 3. The public-API / `ClassifierMixin` layer (mostly mechanical)

`ClassifierGLMHMM` already inherits `ClassifierMixin`. Audit each mixin method
against the `GLMHMM` signatures (which differ from `GLM` — they take
`session_starts`). The mixin was written for `GLM`, so several overrides assume
the `GLM` signature and must be re-pointed at the `GLMHMM` ones.

Methods to provide / verify on `ClassifierGLMHMM`:

1. **`fit`** — already overridden. ✅ Calls `set_classes`, `encode`, delegates.
2. **`score`** — already overridden. ✅
3. **`_preprocess_inputs`** — the mixin version
   ([classifier_glm.py:186-197](../glm/classifier_glm.py#L186-L197)) one-hot encodes
   `y`. Confirm `GLMHMM._preprocess_inputs` has a compatible signature
   (`drop_nans`) and that one-hot happens *after* NaN dropping. This is where
   integer `y → (n_time, n_classes)` one-hot conversion must land for the EM loop.
4. **`predict` / `predict_proba`** — `GLMHMM.predict` returns state-marginalized
   rates. For categorical these are per-class log-probabilities of shape
   `(n_time, n_classes)`. Override to `argmax`/decode (mirror
   [classifier_glm.py:203-293](../glm/classifier_glm.py#L203-L293)) — but verify
   the GLMHMM `predict` axis semantics, since classes are at `axis=-2` *before*
   state marginalization and `axis=-1` *after*.
5. **`simulate`** — `GLMHMM.simulate` has a richer signature
   (`state_format`, `session_starts`) and returns
   `(activity, rates, states)`, not `(y, log_prob)`. The mixin's `simulate`
   override ([classifier_glm.py:364-413](../glm/classifier_glm.py#L364-L413))
   assumes the `GLM` return shape — it must be re-written for the GLMHMM return
   tuple, decoding the one-hot `activity` back to labels.
6. **`update` / `initialize_optimizer_and_state` / `initialize_params` /
   `compute_loss`** — the mixin overrides encode `y` and call `super()`. Confirm
   each lines up with the GLMHMM signature (`session_starts`).
7. **`_estimate_resid_degrees_of_freedom`** — the mixin version
   ([classifier_glm.py:295-362](../glm/classifier_glm.py#L295-L362)) infers
   `n_neurons` from `coef.ndim` (2 vs 3). For GLMHMM `coef` is `(n_features,
   n_classes, n_states)` (ndim 3) — the inference logic will misread the class
   axis as the neuron axis. This needs a GLMHMM-specific override.

> Action: walk every `ClassifierMixin` method and decide *inherit / override /
> no-op*. The ones that assume the `GLM` (no `session_starts`) signature or the
> `GLM` return shapes (`predict`, `simulate`, `_estimate_resid_degrees_of_freedom`)
> are the ones that need GLMHMM-aware overrides.

---

## 4. The core EM changes (the hard part)

These are the changes that cannot be done from the subclass alone, because the
shared EM machinery assumes "states are last" and conflates "has an extra axis"
with "is a population to sum over."

### 4.1 `log_softmax` is applied over the wrong axis

`inverse_link_function` is applied to the **full** rate tensor inside
`compute_rate_per_state` ([utils.py:59](utils.py#L59)):

```python
predicted_rate_given_state = inverse_link_function(lin_comb + intercept)
```

For categorical the full tensor is `(n_time, n_classes, n_states)`. The default
`log_softmax` (`jax.nn.log_softmax`, `axis=-1`) would normalize over **states**,
not classes. We need normalization over the **class axis** (`axis=-2`).

**Fix:** `ClassifierGLMHMM` must default `inverse_link_function` to
`log_softmax` applied over the class axis, e.g.
`functools.partial(jax.nn.log_softmax, axis=-2)` (wrapped so it still resolves
through `resolve_inverse_link_function`). Do **not** rely on
`CategoricalObservations.default_inverse_link_function`, which is the `axis=-1`
version correct only when classes are last (as in `ClassifierGLM`).

> Verify the population/`predict` path too: after state marginalization the
> class axis becomes last again, so any post-marginalization softmax/normalization
> must use the right axis at that point.

### 4.2 E-step wrongly sums over the state axis

`prepare_estep_log_likelihood`
([algorithm_configs.py:232-239](algorithm_configs.py#L232-L239)):

```python
def log_likelihood(params, X, y):
    rate = compute_rate_per_state(X, params, inverse_link_function)
    scale = jnp.exp(params.log_scale) if params.log_scale is not None else None
    log_like = log_likelihood_per_sample(y, rate, scale)
    if is_population_glm:
        log_like = log_like.sum(axis=1)   # <-- sum over neurons
    return log_like
```

For categorical, `log_likelihood_per_sample` already consumes the class axis
inside the categorical NLL (`axis=-1`), so its output is `(n_time, n_states)`.
The `is_population_glm` branch then does `.sum(axis=1)`, which collapses the
**state** axis — destroying exactly the per-state likelihoods the E-step needs.

This is the central bug. The `is_population_glm` flag means two different things:
(a) "the rate has an extra leading axis so vmap the state axis at position 2"
(needed and correct for categorical), and (b) "there is an independent neuron
axis to marginalize by summing" (true for population, **false** for categorical —
the class axis is part of a single multinomial likelihood, not independent
neurons).

**Fix options (pick one):**

- **Preferred — make the reduction shape-driven, like the M-step already is.**
  Replace the `if is_population_glm:` post-sum with the same `if log_like.ndim > 2:
  log_like = log_like.sum(axis=1)` guard used in
  `_posterior_weighted_objective_impl`. For categorical the per-state likelihood
  is `(n_time, n_states)` (ndim 2 → no sum); for true population it is
  `(n_time, n_neurons, n_states)` (ndim 3 → sum over neurons). This keeps state
  vmapping driven by `is_population_glm` but stops the spurious state collapse.
- **Alternative — pass an explicit `observation_model`-aware flag**
  (e.g. `aggregate_neurons: bool`) from `_log_likelihood` so the categorical
  model can request "vmap state axis at 2, do not neuron-sum."

Whatever the choice, the `_log_likelihood` cache key
([glm_hmm.py:380-384](glm_hmm.py#L380-L384)) already includes the observation
model, so a categorical-specific branch caches correctly.

### 4.3 Scale must be genuinely fixed (`log_scale`) for categorical

Two inconsistencies around scale:

1. `ClassifierGLMHMMValidator.get_empty_params`
   ([validation.py:219-233](validation.py#L219-L233)) builds a plain `GLMParams`
   (no `log_scale`), but `GLMHMM` everywhere uses `GLMHMMModelParams` with a
   `log_scale` field. `_simulate` does `jnp.exp(params.model_params.log_scale)`
   ([glm_hmm.py:924](glm_hmm.py#L924)) → `exp(None)` crash. Use
   `GLMHMMModelParams(coef, intercept, log_scale=jnp.zeros(...))` so scale is a
   well-formed array of ones (`exp(0)=1`).
2. `CategoricalObservations` has `_separable_scale = True` but **no** analytical
   scale update and `has_fixed_scale` returns `False`. In
   `prepare_mstep_update_fn`
   ([algorithm_configs.py:448-537](algorithm_configs.py#L448-L537)) this routes
   into the *numerical* scale-optimization branch — pointless work on a scale that
   has no effect on a `log_softmax` likelihood, and a potential source of
   instability.

   **Fix:** make scale fixed for categorical. Cleanest is to add
   `CategoricalObservations` to the `has_fixed_scale` set
   ([algorithm_configs.py:27-29](algorithm_configs.py#L27-L29)) (it behaves like
   Poisson/Bernoulli: scale is a constant, not a free parameter). Then the M-step
   skips scale optimization entirely and the E-step uses the fixed-scale vmap.

---

## 5. Initialization details to verify

- **Intercept init.** `random_glm_params_init` initializes the intercept from the
  mean rate. Under `log_softmax`, the "mean rate" should correspond to log class
  frequencies (so the initial model predicts the marginal class distribution).
  Confirm the existing mean-rate logic produces a sensible value in log-prob space
  for one-hot `y`, or supply a categorical-specific intercept init.
- **KMeans init.** `KMeansInitializerGLM`
  ([initialize_parameters.py:116-248](initialize_parameters.py#L116-L248))
  fits a per-state `GLM`/`PopulationGLM`. For categorical it must fit a
  `ClassifierGLM`/`ClassifierPopulationGLM` instead, and `scale()` must respect
  the fixed scale. If KMeans init is out of scope for v1, restrict
  `glm_params_init`/`scale_init` to the `"random"`/`"constant"` strategies and
  raise a clear error for `"kmeans"`.

---

## 6. Validator follow-ups

`ClassifierGLMHMMValidator` exists but is incomplete:

- `expected_param_dims` / `check_array_dimensions` are inherited from
  `GLMHMMValidator` and still describe `coef.ndim == 2`. Categorical `coef` is
  ndim 3 (`(n_features, n_classes, n_states)`); update `expected_param_dims` and
  the error-message format string
  ([validation.py:50-88](validation.py#L50-L88)).
- `validate_consistency`
  ([validation.py:111-129](validation.py#L111-L129)) checks
  `log_scale.shape == intercept.shape`. With the fixed-scale approach decide the
  canonical `log_scale` shape (likely `(n_states,)` or broadcastable) and align
  the check, or relax it for categorical.
- Add a `check_and_cast_y` path that accepts integer labels and/or one-hot `y`
  consistent with the mixin's `_preprocess_inputs`.

---

## 7. Suggested implementation order

1. **Scale fixed** — add `CategoricalObservations` to `has_fixed_scale`; fix
   `get_empty_params` to emit `GLMHMMModelParams` with `log_scale` zeros. (§4.3)
2. **E-step reduction** — make the neuron-sum shape-driven (`ndim > 2`). (§4.2)
3. **Link axis** — default `inverse_link_function` to class-axis `log_softmax`. (§4.1)
4. **Validator** — fix dims/messages/consistency for the ndim-3 coef. (§6)
5. **Mixin overrides** — `_preprocess_inputs`, `predict`, `predict_proba`,
   `simulate`, `_estimate_resid_degrees_of_freedom` for the GLMHMM signatures /
   return shapes. (§3)
6. **Init** — verify intercept under `log_softmax`; gate or adapt KMeans. (§5)
7. **Tests** — round-trip on synthetic 2- and 3-class data: `fit` runs EM to
   convergence; `coef_.shape == (n_features, n_classes, n_states)`;
   `predict` returns decoded labels; `predict_proba` rows sum to 1 over classes;
   `simulate` returns labels in `classes_`; string labels round-trip; compare a
   1-state `ClassifierGLMHMM` against `ClassifierGLM` (should match); E-step
   log-likelihood is monotonically non-decreasing across EM iterations
   (catches the §4.2 bug directly).

---

## 8. Quick reference — files touched

| Concern                          | File                                              |
| -------------------------------- | ------------------------------------------------- |
| Public class / mixin overrides   | [classifier_glm_hmm.py](classifier_glm_hmm.py)    |
| E-step neuron-sum, fixed-scale   | [algorithm_configs.py](algorithm_configs.py)      |
| Rate / link application          | [utils.py](utils.py)                              |
| Validator (dims, scale, y)       | [validation.py](validation.py)                    |
| Param container / `log_scale`    | [params.py](params.py)                            |
| Intercept / KMeans init          | [initialize_parameters.py](initialize_parameters.py) |
| Categorical NLL / link / scale   | [../observation_models.py](../observation_models.py) |
| Mixin reference implementation   | [../glm/classifier_glm.py](../glm/classifier_glm.py) |