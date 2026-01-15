# NeMoS Publication Plan 2026

## Publication Target

**Journal target**: Nature Methods, eLife?

**Questions for discussion:**
- [ ] **Submission deadline**: When do we want to submit? (Options: Q2 2026, Q3 2026, Q4 2026)
  - Q4?
- [ ] **Manuscript preparation timeline**: How much time do we need for writing after features are complete?
- [ ] **Publication story**: What is the core narrative?
    - [ ] "NeMoS: A modular framework for neural encoding models" <- favourite so far
    - [ ] "NeMoS: A comprehensive GLM toolkit for neuroscience"
    - [ ] "NeMoS: Flexible statistical modeling framework for neural data"
    - [ ] Other: _________________
- [ ] Point of emphasis: GPU, ecosystem compatible, modular, neural encoding, scalability, feature construction.
- [ ] **Example datasets**: What datasets should we showcase?
  - Head-direction (docs tutorial)? feature construction, functional connectivity, pynapple integration
  - IBL ViSp (Camila's tutorial)?
  - IBL Ashwood (For GLM-HMM)? behavior, choice
  - Vision/sensory? Which one?
    - Space/time separable GLM, fit retina LGN dataset, receptive field estimation
- [ ] **Authors:**
  - Should include major contributors. Major contribution includes: doc pages, source code, and insights and planning discussions.
      - neuroRSE
      - Alex and Eero
      - Bence, Wolf, Camila (IBL dataset + docs GLMHMM), Arina (if we decide to include the PPGLM)

---

## Current Package Status

### Ready for publication:
- GLM module
- Basis system with compositional algebra
- Observations: Poisson, Gamma, Gaussian<sup>1</sup>, NegativeBinomial<sup>2</sup>, Bernoulli
- FeaturePytree
- Regularization: Ridge, Lasso, GroupLasso, ElasticNet

**Open question on observations:**
1. [ ] Do we need to have a least-square solver for Linear Gaussian with Ridge or UnRegularized (and smoothing penalty when we have it)?
2. [ ] Do we want to optimize the scale parameter jointly for NegativeBinomial? If so, what interface?

### Near completion:
- Categorical observation (ready, needs GLM validation adjustment)

---

## Features In Development

### 1. GLM-HMM

**Added value:**
- State-dependent neural encoding models
- Captures latent behavioral/neural state transitions
- Behavior (choice) modeling
- Differentiator vs standard GLM packages

**Current status**: Class designed, needs review and merge

**Remaining work**:
- Review and merge class design
- Documentation:
  - Reproduce Fig 2 from Ashwood paper
  - Background note on initialization
  - How-to guide on customizing initialization
- Improve initialization: add k-means initialization

**Discussion items:**
- [ x ] **Priority**: Essential for publication / Nice to have / Post-publication?
  - Essential
- [ ] **Timeline**: When can review/merge be completed? class ~3 months development, ~1 month docs.
- [ ] **Ownership**: Who owns documentation? Camila
- [ ] **Ownership**: Who owns k-means initialization? (not needed for the paper) Camila/Edoardo
- [ ] **Scope question**: Is k-means initialization required for publication or can it come later?
- [ ] **Demo requirements**: Is Ashwood Fig 2 reproduction sufficient or do we need additional examples? Needed for the paper

---

### 2. PGAM (Penalized Generalized Additive Models)

Note: ping Savin for the hiring of Bence
Current contract until end of the month. Start with the germa

**Added value:**
- Automatic variable selection without combinatorial explosion
- Uncertainty estimates: approximate confidence intervals on filters
- Interpretable non-linear effects

**Current status**: Main algorithms ported to JAX, Bence should be allocating time

**Requirements**:
- Smoothing penalty (2 kinds):
  1. Difference-based penalty
  2. Derivative-based penalty
- Basis derivative computation
- Integration with existing regularization framework

**Discussion items:**
- [ x ] **Priority**: Essential for publication / Nice to have / Post-publication?
  - Essential (or Nice to have)
- [ ] **Bence's allocation**: How much time can Bence realistically allocate? 10h/week SF, 20/week German contract
- [ ] **Timeline**: Completion date if included? _________________
- [ ] **Scope question**: Do we need both penalty types for publication or can we start with one?
- [ ] **Scope question**: Can basis derivatives be scoped as part of PGAM work or separate feature?
- [ ] **Ownership**: Who owns smoothing regularization implementation? _________________
- [ ] **Publication value**: Does PGAM strengthen our story enough to justify the development time?

---

### 3. PPGLM (Point Process GLM)

**Added value:**
- Scalability to large-scale recordings
- Temporal precision for spike timing analysis
- Improved functional connectivity estimates

**Current status**: Arina working on NeMoS compatibility.
- How to integrate it better questions in regular nemos meeting.
- Not a lot of work
- Test code? use the test_base_regressor_subclasses to check if the model is aligned with NeMoS;

**Discussion items:**
- [ ] **Priority**: Essential for publication / Nice to have / Post-publication?
  - Nice to have
- [ ] **Arina's status**: Where is the compatibility work currently? easy to plug in
- [ ] **Timeline**: Completion date if included? end of june
- [ ] **Ownership**: Who supports Arina on this? Arina
- [ ] **Publication value**: Does PPGLM fit our publication story or is it better suited for follow-up work? yes, but we will see if where we are

---

### 4. Categorical Observation

**Added value:**
- Multiple choice / multi-class modeling
- Enables choice behavior analysis

**Current status**: Ready, needs GLM validation adjustment

**Discussion items:**
- [ ] **Priority**: Essential for publication / Nice to have / Post-publication?
- [ ] **Timeline**: When can validation adjustment be completed? _________________
- [ ] **Ownership**: Who owns this adjustment? _________________
- [ ] **Effort estimate**: Person-days required? _________________

---

## Infrastructure & Performance Features

### 5. Batched Optimization Support

**Proposed**: Add `fit_iterator` method for batch processing

**Added value:**
- Scalability to large datasets
- User convenience: no need to write outer loop externally
- Memory-efficient fitting


**Discussion items:**
- [ ] **Priority**: Essential for publication / Nice to have / Post-publication?
  - Nice to have
- [ ] **Value proposition**: Does this enable new use cases or is it primarily convenience?
- [ ] **Timeline**: Completion date if included? 3 weeks
- [ ] **Ownership**: Who implements this? Bence
- [ ] **Scope question**: Is this critical for the types of datasets we'll showcase in the paper? no

---

### 6. Basis System Improvements

**Proposed improvements**:
- Allow JIT compilation of `compute_features`
- Convert BSpline to JAX (currently uses scipy) Needed
- Implement derivative methods for basis (needed for PGAM)

**Added value:**
- Performance: JIT compilation efficiency
- Enables derivative-based penalization (PGAM)
- Direct basis usage in complex model implementations (e.g., as non-linearities in Gaussian Process inference)
- Pure JAX implementation (no scipy dependency for BSpline)


**Discussion items:**
- [ ] **Priority**: Which improvements are essential vs nice-to-have?
  - [ ] JIT compilation: _________________
  - [ ] BSpline JAX conversion: _________________
  - [ ] Derivative methods: _________________
- [ ] **Dependencies**: Derivative methods needed only if PGAM is included?
- [ ] **Timeline**: Completion dates for included features? _________________
- [ ] **Ownership**: Who works on basis improvements? _________________

---

### 7. Regularization

**Proposed improvements**:
- Allow pytree regularizer as an alternative to a single scalar regularizer strength.
- Smoothing penalization

**Added value:**
- Flexible penalization
- Smooth response modeling.
- PGAM integration (per coefficient group-specific regularization needed).

**Discussion items:**
- [ ] **Priority**: Essential / Can wait?
  - Essential
- [ ] **Timeline**: When should this be completed? mid Feb pytree,
- [ ] **Status**: Is the transition essentially complete or is work remaining? pytree almost done
- [ ] **Ownership**: Who handles remaining transition work? Wolf

### 8. Solver Maintenance

**Added value:**
- Maintainability: no need to patch jaxopt port
- Faster CI: reduced testing time by dropping jaxopt tests
- Reliability: eliminates random test failures

**Proposed**: Drop jaxopt completely (tests randomly failing)

**Discussion items:**
- [ ] **Priority**: Essential / Can wait?
  - Essential
- [ ] **Timeline**: When should this be completed? pr open.
- [ ] **Status**: Is the transition essentially complete or is work remaining? Zoom linesearch...
- [ ] **Ownership**: Who handles remaining transition work? Bence

---

## Publication Scope Decision Framework

### Tier 1: Must-Have Features
*Features absolutely required for publication*

**Decision items:**
- [ ] Feature 1: _________________
- [ ] Feature 2: _________________
- [ ] Feature 3: _________________
- [ ] Rationale: _________________

### Tier 2: Should-Have Features
*Features that strengthen publication but aren't dealbreakers*

**Decision items:**
- [ ] Feature 1: _________________
- [ ] Feature 2: _________________
- [ ] Rationale: _________________

### Tier 3: Nice-to-Have Features
*Features for post-publication or if time allows*

**Decision items:**
- [ ] Feature 1: _________________
- [ ] Feature 2: _________________
- [ ] Rationale: _________________

---

## Resource Allocation

### Team capacity assessment:

**Discussion items:**

**External collaborators:**

- [ ] **Arina's availability**:
  - Time? 2/3 months development starting after mid Feb. Goal finish before grad-school.
  - Features? PPGLM (MCMC fitting no control variate, no PolyApprox)
- [ ] **Bence's availability**:
  - Time? Ideally gap between this contract sith SF (until Feb 2nd 2026) and new one ~ 3 months. PGAM on German contract, right away.
  - Features? PGAM, scalability (batching)
- [ ] **Camila's availability**:
  - Time? 10h/week since Feb, ask Allison for extension.
  - Priority: Ashwood reproducing fig 2
  - Features? GLM-HMM docs, _________________
- [ ] **Wolf's availability**:
  - Time? 10h/week split between pynapple, nemos and plenoptic...
  - Features? Regularization pytree, multiplicative basis bounds, smoothing penalty.

Note: I assume neuroRSE have more flexible time allocation.

**NeuroRSE**
- [ ] **Billy's availability**:
  - Features? _________________
- [ ] **Edoardo's availability**:
  - Features? Any of the implementation, paper writing.
- [ ] **Guillaume's availability**:
  - Features? _________________
- [ ] **Sarah's availability**:
  - Features? Basis compilation, _________________

---

## Milestones & Timeline

I need a deadline otherwise I'll never have a paper.

**Key dates** (to be filled in):

- [ ] **Target submission date**: _________________
- [ ] **Draft ready**: _________________
- [ ] **Internal review draft**: _________________
- [ ] **External collaborator review**: _________________
- [ ] **Feature freeze date**: _________________
- [ ] **Code review complete**: _________________
- [ ] **Documentation complete**: _________________

---

## Publication Differentiators

**What makes NeMoS compelling vs alternatives?**

Discussion items:
- [ ] Breadth of models (GLM, GLM-HMM, PGAM, PPGLM)?
- [ ] Flexibility/composability of basis system?
- [ ] JAX backend enabling performance/scalability?
- [ ] Integration with pynapple ecosystem?
- [ ] Ease of use / API design?
- [ ] Other: _________________

**Main differentiator(s)**: _________________


---

## Action Items

Create a GitHub project for the publication and include there all the action items and deadlines.

---

## Next Meeting

- [ ] **Date**: Mid February.
- [ ] **Agenda**: Review progress on action items, adjust timeline if needed
