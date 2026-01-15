# NeMoS Publication Plan 2026

## Publication Target

**Journal target**: eLife or similar

**Questions for discussion:**
- [ ] **Submission deadline**: When do we want to submit? (Options: Q2 2026, Q3 2026, Q4 2026)
- [ ] **Manuscript preparation timeline**: How much time do we need for writing after features are complete?
- [ ] **Publication story**: What is the core narrative?
    - [ ] "NeMoS: A composable framework for neural encoding models"
    - [ ] "NeMoS: A comprehensive GLM toolkit for neuroscience"
    - [ ] "NeMoS: Flexible statistical modeling framework for neural data"
    - [ ] Other: _________________
- [ ] **Example datasets**: What datasets should we showcase?
  - Head-direction (docs tutorial)?
  - IBL ViSp (Camila's tutorial)?
  - IBL Ashwood (For GLM-HMM)?
  - Vision/sensory? Which one?

---

## Current Package Status

### Ready for publication:
- GLM module
- Basis system with compositional algebra
- Observations: Poisson, Gamma, Gaussian<sup>1</sup>, NegativeBinomial<sup>2</sup>, Bernoulli
- FeaturePytree
- Regularization: Ridge, Lasso, GroupLasso

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
- [ ] **Priority**: Essential for publication / Nice to have / Post-publication?
- [ ] **Timeline**: When can review/merge be completed? _________________
- [ ] **Ownership**: Who owns documentation? _________________
- [ ] **Ownership**: Who owns k-means initialization? _________________
- [ ] **Scope question**: Is k-means initialization required for publication or can it come later?
- [ ] **Demo requirements**: Is Ashwood Fig 2 reproduction sufficient or do we need additional examples?

---

### 2. PGAM (Penalized Generalized Additive Models)

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
- [ ] **Priority**: Essential for publication / Nice to have / Post-publication?
- [ ] **Bence's allocation**: How much time can Bence realistically allocate? _________________
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

**Current status**: Arina working on NeMoS compatibility

**Discussion items:**
- [ ] **Priority**: Essential for publication / Nice to have / Post-publication?
- [ ] **Arina's status**: Where is the compatibility work currently? _________________
- [ ] **Timeline**: Completion date if included? _________________
- [ ] **Ownership**: Who supports Arina on this? _________________
- [ ] **Publication value**: Does PPGLM fit our publication story or is it better suited for follow-up work?

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
- [ ] **Value proposition**: Does this enable new use cases or is it primarily convenience?
- [ ] **Timeline**: Completion date if included? _________________
- [ ] **Ownership**: Who implements this? _________________
- [ ] **Scope question**: Is this critical for the types of datasets we'll showcase in the paper?

---

### 6. Basis System Improvements

**Proposed improvements**:
- Allow JIT compilation of `compute_features`
- Convert BSpline to JAX (currently uses scipy)
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

### 8. Solver Maintenance

**Added value:**
- Maintainability: no need to patch jaxopt port
- Faster CI: reduced testing time by dropping jaxopt tests
- Reliability: eliminates random test failures

**Proposed**: Drop jaxopt completely (tests randomly failing)

**Discussion items:**
- [ ] **Priority**: Essential / Can wait?
- [ ] **Timeline**: When should this be completed? _________________
- [ ] **Status**: Is the transition essentially complete or is work remaining? _________________
- [ ] **Ownership**: Who handles remaining transition work? _________________

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

- [ ] **Bence's availability**:
  - Time? _________________
  - Features? PGAM, _________________
- [ ] **Arina's availability**:
  - Time? _________________
  - Features? PPGLM, _________________
- [ ] **Camila's availability**:
  - Time? _________________
  - Features? GLM-HMM docs, _________________

Note: I assume neuroRSE have more flexible time allocation.

**NeuroRSE**
- [ ] **Sarah's availability**:
  - Features? Any of the implementation, paper writing.
- [ ] **Sarah's availability**:
  - Features? Basis compilation, _________________
- [ ] **Guillaume's availability**:
  - Features? _________________
- [ ] **Billy's availability**:
  - Features? _________________


---

## Milestones & Timeline

**Key dates** (to be filled in):

- [ ] **Feature freeze date**: _________________
- [ ] **Code review complete**: _________________
- [ ] **Documentation complete**: _________________
- [ ] **Example notebooks complete**: _________________
- [ ] **Internal review draft**: _________________
- [ ] **External collaborator review**: _________________
- [ ] **Submission ready**: _________________
- [ ] **Target submission date**: _________________

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

## Success Criteria

**What does a successful publication look like?**

- [ ] **Technical contribution**: _________________
- [ ] **Scientific impact**: _________________
- [ ] **Community adoption goals**: _________________
- [ ] **Comparison benchmarks**: What should we compare against? _________________

---

## Action Items

*To be filled in during/after meeting*: create a project for this.

| Action | Owner | Deadline | Status |
|--------|-------|----------|--------|
|        |       |          |        |
|        |       |          |        |
|        |       |          |        |

---

## Next Meeting

- [ ] **Date**: _________________
- [ ] **Agenda**: Review progress on action items, adjust timeline if needed
