(developers-hessian-tagging)=
# Hessian tagging

## Background

A Newton step solves `H d = -g`, and how that linear system is best solved depends on what is known about `H` beforehand. A Cholesky factorization is the cheapest option but needs a positive definite matrix and fails on a singular one. A pseudo-inverse handles a singular matrix but costs more. A diagonal Hessian can be solved in time linear in the number of parameters. And if `H` has no sign guarantee at all, `-H^{-1} g` need not even point downhill, in which case a Newton step is the wrong thing to take and a first-order or quasi-Newton solver is a better choice.

Checking any of this at run time means factorizing the matrix, which costs as much as the Newton solve itself. Under `jit` a failed Cholesky cannot be caught and retried either. So the information is carried alongside the Hessian instead, as a `HessianTag` that each model and each regularizer declares about the term it contributes. The tags are combined when the terms are added, and the result picks the linear solver.

What the tag has to distinguish follows from what the decisions turn on:

| decision | turns on                                                                          |
| --- |-----------------------------------------------------------------------------------|
| Newton step versus a first-order solver | whether `H` is positive semidefinite (PSD); otherwise the step can point uphill   |
| Cholesky versus a pseudo-inverse | whether `H` is invertible, i.e. definite (D) rather than merely semidefinite (SD) |
| a diagonal solve, linear in the number of parameters | whether `H` is diagonal                                                           |
| one factorization per neuron | whether `H` is block diagonal, one block per neuron                               |


This note derives a tagging system that requires only structural properties of a symmetric hessian matrix, and a decision rule that assigns a coherent sign (none, PSD or PD) to the sum of the hessians based exclusively on their tags. The discussion will proceed as follows:

1. We will prove that no tag-based rule can be complete: the sum of two matrices with identical spectrum can differ in sign.
2. We will propose a sound tagging system based on: the guaranteed sign of the hessian (example: a GLM with canonical link is guaranteed PSD); the largest PD principal block of the hessian (could be empty or the whole hessian); the largest flat principal block. Here a block corresponds to a union of parameter leaves.
3. We will derive an optimal, sound rule for the chosen tagging system.
4. We will discuss why the proposed tagging system has the desired granularity (leaf based), and why any finer tag would require (at least for GLMs) computationally heavy characterizations of the design matrix.

## Problem Setting: combining two Hessians

In nemos a model is fit by minimizing a loss with an additive penalty,

$$\mathcal{L}(\beta) = \ell(\beta) + p_\lambda(\beta),$$

so what the solver actually differentiates twice is a sum, and its Hessian is the sum of the two Hessians,

$$\nabla^2 \mathcal{L}(\beta) = \nabla^2 \ell(\beta) + \nabla^2 p_\lambda(\beta).$$

The two terms are declared by different objects. A model knows the curvature of its likelihood, a regularizer knows the curvature of its penalty, and neither has seen the other. The decisions in the table above are about the sum, which no single object is in a position to describe.

If the definiteness of the sum followed from the definiteness of the two terms, this would be a small problem: each term would carry one label, and putting them together would be a lookup table. It does not follow, and the case that breaks it is the most ordinary one nemos has.

Take a GLM whose parameters split into $p$ coefficients and an intercept, $\beta = (\beta_{\text{coef}}, \beta_{\text{int}}) \in \mathbb{R}^{p+1}$, penalized by a ridge term that leaves the intercept alone,

$$p_\lambda(\beta) = \frac{\lambda}{2} \lVert \beta_{\text{coef}} \rVert^2, \qquad \nabla^2 p_\lambda = \begin{pmatrix} \lambda I & 0 \\ 0 & 0 \end{pmatrix}.$$

That Hessian is positive semidefinite and singular: the penalty does not depend on the intercept, so it has no curvature there at all. The loss Hessian is

$$\nabla^2 \ell(\beta) = \tilde{X}^\top W(\beta) \tilde{X}, \qquad \tilde{X} = \begin{pmatrix} X & \mathbf{1}\end{pmatrix}, \qquad W(\beta) = \operatorname{diag}\big(w_1(\beta), \dots, w_T(\beta)\big),$$

with $w_t(\beta) \ge 0$ the curvature contributed by sample $t$, which depends on the parameters through the linear predictor. It is positive semidefinite as well, and singular whenever the design is rank deficient, which a basis summing to one plus an intercept column already is.

So both terms are labelled positive semidefinite and both are singular, and from those two labels nothing whatsoever follows about the sum. The sum is nevertheless positive definite, and the reason is a statement about null spaces. For positive semidefinite $A$ and $B$, $v^\top (A + B) v = 0$ forces each quadratic form to vanish separately, so

$$\operatorname{null}(A + B) = \operatorname{null}(A) \cap \operatorname{null}(B).$$

Here $\operatorname{null}(\nabla^2 p_\lambda) = \operatorname{span}\{e_{p+1}\}$, the intercept axis, whereas $e_{p+1} \notin \operatorname{null}(\nabla^2 \ell(\beta))$, because $e_{p+1}^\top \nabla^2 \ell(\beta) \, e_{p+1} = \mathbf{1}^\top W(\beta) \mathbf{1} = \sum_t w_t(\beta) > 0$. The two null spaces are subspaces of $\mathbb{R}^{p+1}$ intersecting only at the origin, so the sum has trivial null space and is positive definite.

What a tag has to carry, beyond its own definiteness, is therefore enough to locate those two subspaces and decide whether they meet. It cannot locate an arbitrary subspace: the fields defined below are sets of leaves, so the only subspaces they can identify are those spanned by whole leaves. That is enough for this example, where the penalty's null space is exactly the intercept leaf, and it is what the next section makes precise; the example comes back once the fields are defined.

## Notation

Everything from here on uses these symbols.

| symbol | meaning |
| --- | --- |
| $\beta \in \mathbb{R}^N$ | the parameters: a pytree with $n$ leaves, $N$ scalar entries in total |
| $\ell$, $p_\lambda$, $\mathcal{L} = \ell + p_\lambda$ | loss, penalty, and the objective that is minimized |
| $\lambda$ | the regularization strength: a scalar, or an array with one entry per parameter |
| $H = \nabla^2 \mathcal{L}(\beta)$ | the Hessian a tag describes, itself a function of $\beta$ |
| $i$, $I_i$ | a leaf, and the coordinates it occupies; the $I_i$ partition $\{1, \dots, N\}$ |
| $\beta^{(S)}$ | the parameters of the leaves in a set $S$, concatenated into one vector |
| $H[S]$ | the block $\partial^2 \mathcal{L} / \partial \beta^{(S)} \partial \beta^{(S)\top}$, the principal submatrix of $H$ on those coordinates |
| $V_i$, $V_S$ | the coordinate subspace $\operatorname{span}\{e_j : j \in I_i\}$ of leaf $i$, and $\bigoplus_{i \in S} V_i$ |
| $\operatorname{Sym}(N)$ | the real symmetric $N \times N$ matrices |
| $\operatorname{null}(A)$ | the null space $\{v : Av = 0\}$ |
| $A \succeq 0$, $A \succ 0$ | positive semidefinite, positive definite |

Every matrix here is symmetric. The Hessian of a twice continuously differentiable objective is, by Schwarz's theorem, and the Fisher and Gauss-Newton approximations are by construction, so $\operatorname{Sym}(N)$ is the right ambient set and symmetry is used without further comment.

$H[S]$ is the curvature seen when only the parameters of $S$ move, everything else held fixed. Reordering the parameters so that $S$ comes first turns it into a diagonal block, and a permutation changes no eigenvalue, so picturing these sets as diagonal blocks is exact rather than a convenient approximation.

Restriction preserves the quadratic form on vectors that live in the block. Writing $v[S]$ for the restriction of $v$ to the coordinates of $S$, if $v \in V_S$ then

$$v^\top H v = \sum_{j, k \in I_S} v_j H_{jk} v_k = v[S]^\top H[S] \, v[S] ,$$

since every term the restriction drops carries a zero entry of $v$. In particular $e_j[S]$ is a standard basis vector of $\mathbb{R}^{d(S)}$ and $e_j[S]^\top H[S] \, e_j[S] = H_{jj}$.

## What a block-spectral description can decide

The decision table has to be answered without looking at the matrix, so the only thing available about each term is a description of it, fixed in advance. Before designing one it is worth knowing the ceiling.

:::{admonition} Definition (block-spectral description, sound and complete rules)
:class: note

Let $\Sigma = \{\text{none}, \ {\succeq}\,0, \ {\succ}\,0\}$, each element a predicate on $\operatorname{Sym}(N)$, and write $M \models \sigma$ when $M$ satisfies it. Order $\Sigma$ by implication,

$$\sigma' \sqsubseteq \sigma \iff \big(M \models \sigma \implies M \models \sigma'\big) \quad \text{for all } M \in \operatorname{Sym}(N) ,$$

which totally orders it as $\text{none} \sqsubset {\succeq}\,0 \sqsubset {\succ}\,0$. Only the positive signs are carried: a claim that $H$ is negative (semi)definite is the same claim about $-H$, and $\nabla^2 \mathcal{L}$ is the Hessian of something being minimized. The order gives a unique strongest true sign

$$\sigma : \operatorname{Sym}(N) \to \Sigma, \qquad \sigma(M) = \max_{\sqsubseteq} \, \{ s \in \Sigma : M \models s \} .$$

For $M$ symmetric of size $k$ let $\operatorname{spec} M \in \mathbb{R}^k$ be its eigenvalues in non-decreasing order, multiplicities included. The *block-spectral description* of $H \in \operatorname{Sym}(N)$ is the function on sets of leaves

$$\tau(H)(S) = \operatorname{spec} H[S] \in \mathbb{R}^{d(S)}, \qquad S \subseteq \{1, \dots, n\} ,$$

where

$$d(S) = \sum_{i \in S} \lvert I_i \rvert$$

counts the scalar parameters those leaves hold, not the leaves. Put

$$\mathcal{T} = \{ \tau(H) : H \in \operatorname{Sym}(N) \} .$$

A *combination rule* is a map $R : \mathcal{T} \times \mathcal{T} \to \Sigma$. It is **sound** if

$$R\big(\tau(H_1), \tau(H_2)\big) \sqsubseteq \sigma(H_1 + H_2) \qquad \text{for all } H_1, H_2 \in \operatorname{Sym}(N),$$

and **complete** if equality holds for all $H_1, H_2 \in \operatorname{Sym}(N)$.

:::

:::{admonition} Theorem 1 (no combination rule is complete)
:class: important

Let $N \ge 2$. Then there is no $R : \mathcal{T} \times \mathcal{T} \to \Sigma$ with $R\big(\tau(H_1), \tau(H_2)\big) = \sigma(H_1 + H_2)$ for all $H_1, H_2 \in \operatorname{Sym}(N)$.

:::

:::{admonition} Proof
:class: dropdown

Fix two coordinates $j_1 \neq j_2$, which exist because $N \ge 2$, and let $P = \operatorname{span}\{e_{j_1}, e_{j_2}\}$. Set

$$u = \tfrac{1}{\sqrt{2}}\big(e_{j_1} - e_{j_2}\big), \qquad v = \tfrac{1}{\sqrt{2}}\big(e_{j_1} + e_{j_2}\big) ,$$

so $\lVert u \rVert = \lVert v \rVert = 1$ and $u^\top v = \tfrac{1}{2}(1 - 1) = 0$: the pair is an orthonormal basis of $P$, hence $u u^\top + v v^\top = I_P$. Put

$$A = I - u u^\top, \qquad B = I - v v^\top, \qquad A' = B' = A .$$

$A$ and $B$ are symmetric, and $u^\top u = v^\top v = 1$ with $u^\top v = 0$ give

$$A u = 0, \quad A v = v, \qquad B v = 0, \quad B u = u .$$

So $A$ kills $u$ and is the identity on $u^\perp$, while $B$ kills $v$ and is the identity on $v^\perp$: each is positive semidefinite with a one-dimensional null space, hence singular. Since $u u^\top + v v^\top$ acts as the identity on $P$ and as zero on $P^\perp$,

$$A + B = 2I - \big(u u^\top + v v^\top\big) = I_P \oplus 2 I_{P^\perp} \succ 0 ,$$

so $\sigma(A + B) = {\succ}\,0$, while $A' + B' = 2A$ is positive semidefinite and singular, so $\sigma(A' + B') = {\succeq}\,0$.

The two have the same block-spectral description, because $B$ is $A$ conjugated by a coordinate sign flip. Let $R$ be the diagonal matrix with $-1$ in position $j_2$ and $1$ elsewhere. Then $R u = v$, so

$$B = I - (Ru)(Ru)^\top = R \big(I - u u^\top\big) R^\top = R A R^\top .$$

$R$ is diagonal, so it scales coordinates without mixing them and restriction commutes with the conjugation: $B[S] = R[S] \, A[S] \, R[S]^\top$ for every $S$. Each $R[S]$ is again diagonal with entries $\pm 1$, hence orthogonal, and orthogonal similarity preserves spectra. So $\operatorname{spec} B[S] = \operatorname{spec} A[S]$ for every $S$, that is $\tau(A) = \tau(B)$, hence $\tau(A) = \tau(A')$ and $\tau(B) = \tau(B')$.

A complete $R$ would satisfy $R(\tau(A), \tau(B)) = \sigma(A + B) = {\succ}\,0$ and $R(\tau(A'), \tau(B')) = \sigma(A' + B') = {\succeq}\,0$. The two left-hand sides are the value of $R$ at the same argument, and ${\succ}\,0 \neq {\succeq}\,0$.

:::

The hypothesis $N \ge 2$ is sharp. For $N = 1$ every $H = (h)$ has $\tau(H)$ recording $\operatorname{spec} H[S] = (h)$ on the leaf holding that coordinate, so $\tau$ is injective and $R\big(\tau(h_1), \tau(h_2)\big) = \sigma(h_1 + h_2)$ is a well defined complete rule.

The two configurations differ only in the relative position of the two null spaces, $\operatorname{span}\{u\}$ against $\operatorname{span}\{v\}$, and no description built from sets of leaves records that. Recovering it means locating null spaces. That is spectral information, and computing it costs a factorization of the same order as the Newton solve.

What Theorem 1 rules out is a complete rule: no map from two descriptions to a sign returns $\sigma(H_1 + H_2)$ in every case. A sound rule is still possible. Its answer is always true of the sum, and sometimes weaker than the truth. The rest of this note builds one that reads only structural properties of the two terms and holds at every $\beta$.

The two mistakes do not cost the same. Refusing to certify a matrix that is definite only makes the linear solve slower. Certifying one that is not gives a wrong step or a NaN, silently and under `jit`.

## Assumptions the tagging system rests on

Everything below is derived under five assumptions, collected here before any of the machinery and stated without reference to it. Each is used later, and together they are what makes the combination rule sound: whatever a combined tag claims is true of the sum it describes.

:::{admonition} Assumption 1 (additive objective)
:class: note

$$\mathcal{L}(\beta) = \ell(\beta) + p_\lambda(\beta), \qquad \nabla^2 \mathcal{L}(\beta) = \nabla^2 \ell(\beta) + \nabla^2 p_\lambda(\beta).$$

:::

:::{admonition} Assumption 2 (the strength multiplies the penalty, index by index)
:class: note

$$p_\lambda(\beta) = \sum_i \lambda_i \, p^{(i)}\big(\beta^{(i)}\big), \qquad \nabla^2 p_\lambda(\beta) = \bigoplus_i \lambda_i \nabla^2 p^{(i)}\big(\beta^{(i)}\big),$$

with each $p^{(i)}$ convex and depending only on $\beta^{(i)}$.

:::

A single scalar strength is the case $\lambda_i = \lambda$ for all $i$. An array strength has to be contracted against the index it is indexed by, which is what this sum is; nothing else would leave $p_\lambda$ a scalar.

:::{admonition} Assumption 3 (non-negative strength)
:class: note

$$\lambda_i \ge 0 \quad \text{for every } i.$$

:::

A negative entry makes the penalty curve downwards there, and a positive semidefinite matrix plus a negative semidefinite one carries no sign at all.

:::{admonition} Assumption 4 (statements are quantified over all $\beta$)
:class: note

Every statement made below about $\nabla^2 \mathcal{L}$ holds at every $\beta$ in the parameter space, not at the optimum $\hat\beta$ alone.

:::

$\nabla^2 \mathcal{L}$ is a matrix-valued function of the parameters, and a description of it is fixed once, before the fit, then relied on at every step; one that held only near the solution would be wrong where the solver starts. For the GLM loss of the previous section this is the difference between $\sum_t w_t(\beta) > 0$ for all $\beta$, which is true for a link that keeps the likelihood convex, and the same inequality at the solution alone, which would be worth nothing.

:::{admonition} Assumption 5 (leaf resolution)
:class: note

Every statement a tag makes about $H$ is evaluated on the block of a whole set of leaves,

$$\operatorname{flat}(S) \iff H[S] = 0, \qquad \operatorname{definite}(S) \iff H[S] \succ 0,$$

and a tag says nothing about a proper subspace of a leaf.

:::

A leaf whose block is neither zero nor definite is therefore left out of both sets. What the two sets then say about $\operatorname{null}(H)$ is the following.

:::{admonition} Proposition 1 (a tag bounds the null space, and bounds it exactly only when the null space is leaf-aligned)
:class: important

Let $F$ and $D$ be the sets of leaves a tag records as flat and as definite, so that $H[F] = 0$ and $H[D] \succ 0$. If $H$ is symmetric with $H \succeq 0$, then

$$V_F \subseteq \operatorname{null}(H), \qquad V_D \cap \operatorname{null}(H) = \{0\}.$$

If moreover $F$ is the largest set of leaves whose block vanishes, then $V_F = \operatorname{null}(H)$ if and only if $\operatorname{null}(H) = V_S$ for some set of leaves $S$.

:::

:::{admonition} Proof
:class: dropdown

Let $j \in I_F$, let $k \in \{1, \dots, N\}$ and let $t \in \mathbb{R}$. Since $H[F] = 0$, in particular $H_{jj} = 0$, and since $H \succeq 0$ the quadratic form $x^\top H x$ is non-negative at every $x$. Evaluating it at $x = e_j + t e_k$,

$$0 \le (e_j + t e_k)^\top H (e_j + t e_k) = H_{jj} + t \, (H_{jk} + H_{kj}) + t^2 H_{kk} = t \, (2 H_{jk} + t H_{kk}) ,$$

the last step by symmetry. This holds for every $t$. Dividing by $t > 0$ and letting $t \to 0^+$ gives $H_{jk} \ge 0$; dividing by $t < 0$, which reverses the inequality, and letting $t \to 0^-$ gives $H_{jk} \le 0$. Hence $H_{jk} = 0$, and by symmetry of the Hessian $0 = H_{jk} = H_{kj}$. As $k$ was arbitrary, $H e_j = 0$, and as $j \in I_F$ was arbitrary this holds for every basis vector of $V_F$; by linearity every $v = \sum_{j \in I_F} v_j e_j$ satisfies

$$H v = \sum_{j \in I_F} v_j \, H e_j = 0 ,$$

so $V_F \subseteq \operatorname{null}(H)$.

To prove the second claim, let $v \in V_D \cap \operatorname{null}(H)$. From $Hv = 0$ we get $v^\top H v = 0$, and the same expansion, now with $v$ supported on $I_D$, leaves $v^\top H v = v_D^\top H[D] \, v_D$. Since $H[D] \succ 0$ this forces $v_D = 0$, hence $v = 0$. No assumption on the sign of $H$ is used here.

To prove the last claim, suppose $\operatorname{null}(H) = V_S$ for a set of leaves $S$. Every $v \in V_S$ has $H v = 0$, so $u^\top H v = 0$ for all $u, v \in V_S$. Taking $u = e_i$ and $v = e_j$ with $i, j \in I_S$ gives $H[S]_{ij} = e_i^\top H e_j = 0$, so $H[S] = 0$. By maximality $S \subseteq F$, hence $V_S \subseteq V_F$, and combining with the first inclusion, $V_S \subseteq V_F \subseteq \operatorname{null}(H) = V_S$. Hence $V_F = \operatorname{null}(H)$. The converse is immediate, since $V_F$ is by construction spanned by whole leaves.

:::

The sign hypothesis cannot be dropped from the first inclusion. Take two leaves of one coordinate each and

$$H = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},$$

which is symmetric with eigenvalues $\pm 1$, so $H \not\succeq 0$. Its block on the first leaf vanishes, $H[\{1\}] = 0$, so that leaf is flat and $V_F = \operatorname{span}\{e_1\}$. But $H e_1 = e_2 \neq 0$, so $e_1 \notin \operatorname{null}(H)$ and $V_F \not\subseteq \operatorname{null}(H)$.

So a tag never claims more about the null space than is true. It is weaker than the truth when the null space cuts across a leaf.

That is what a mixed strength array does. Ridge has one term per coordinate, $p_\lambda(\beta) = \tfrac{1}{2}\sum_j \lambda_j \beta_j^2$ with $\nabla^2 p_\lambda = \operatorname{diag}(\lambda)$, so a leaf's block is zero when every $\lambda_j$ on it is zero, positive definite when every $\lambda_j$ is strictly positive, and singular without being zero when they are mixed. Take coefficients on one leaf of three with $\lambda = (0, 2, 0)$ and an unpenalized intercept: $\operatorname{null}(\nabla^2 p_\lambda)$ has dimension three, while the only leaf that can be recorded as flat is the intercept, so $V_F$ has dimension one. The containment is strict, and no set of leaves names the missing directions. A uniformly positive array behaves exactly like a positive scalar, and a negative one is excluded by Assumption 3, so the mixed case is the only one where anything is lost.

Assumption 5 is about the description, while Assumption 2 is about the penalty, and their indices need not agree: a penalty may be written coordinate by coordinate even though only leaf-level statements are recorded about its Hessian. Restricting to leaves is what keeps the checks cheap, since they are then set operations over leaves rather than over parameters.

One caveat on reading Assumption 5. Both predicates are evaluated on a block, not coordinate by coordinate. For ridge the two coincide, because $\nabla^2 p_\lambda$ is diagonal and $\operatorname{definite}(\{i\})$ reduces to $\lambda_j > 0$ for every $j \in I_i$. In general it does not: $[[1, 1], [1, 1]]$ has both diagonal entries positive and is singular, so definiteness of a block is not definiteness of its parts.

Under these five, the rule given below returns a property that is true of the sum, which is what Theorem 2 states and proves. Every check it performs is set arithmetic over leaves and comparison of integers; no matrix is ever inspected.

One thing is deliberately out of scope. These are statements in exact arithmetic, and a claim true of $\nabla^2 \mathcal{L}(\beta)$ can still fail in floating point, as it does when the per-sample curvatures of a saturated Bernoulli GLM underflow and $\sum_t w_t(\beta)$ becomes exactly zero. No tag prevents that, and none is meant to; guarding against it belongs to how the solver accepts a step.

## Tag definition

:::{admonition} Definition (tag, satisfaction)
:class: note

A *tag* is a triple

$$t = (\sigma, F, D) \in \mathbb{T} = \Sigma \times \mathcal{P}\big(\{1, \dots, n\}\big) \times \mathcal{P}\big(\{1, \dots, n\}\big) .$$

A matrix $H \in \operatorname{Sym}(N)$ *satisfies* $t$, written $H \models t$, when

$$H \models \sigma, \qquad H[F] = 0, \qquad H[D] \succ 0 .$$

A tag is *realizable* if $H \models t$ for some $H$, and *covering* if $F \cup D = \{1, \dots, n\}$.

:::

A tag is declared by the object that contributes the term; nothing computes it from the matrix. Many matrices satisfy the same tag, and one matrix satisfies many tags: `none` with $F = D = \varnothing$ is satisfied by everything. So the field $\sigma$ need not be the strongest true sign, and $F, D$ need not be the largest sets available. Under-claiming is always allowed, which is why a model can declare a tag without inspecting its own Hessian.

:::{admonition} Proposition 2 (satisfaction is determined by the block spectra)
:class: important

If $\tau(H) = \tau(H')$ then $H \models t \iff H' \models t$ for every tag $t$.

:::

:::{admonition} Proof
:class: dropdown

Each of the three conditions in $H \models t$ is a function of $\tau(H)$.

The sign: $H \models \text{none}$ always; $H \models {\succeq}\,0$ iff $\min \tau(H)(\{1, \dots, n\}) \ge 0$; $H \models {\succ}\,0$ iff that minimum is $> 0$.

Flatness: $H[F]$ is symmetric, so it is diagonalisable and vanishes iff all its eigenvalues do, that is iff $\tau(H)(F) = 0$.

Definiteness: $H[D] \succ 0$ iff $\min \tau(H)(D) > 0$.

Each right-hand side depends on $H$ only through $\tau(H)$, so equal descriptions give equal answers.

:::

So a tag is a coarsening of $\tau$: whatever $\tau$ cannot separate, no tag separates either. In particular the two matrices $A$ and $B$ built in the proof of Theorem 1 satisfy exactly the same tags, while $A + B \succ 0$ and $A + A$ is singular. Any tagging scheme that assigns tags on the basis of block spectra therefore inherits the ceiling: it hands the same pair of tags to two configurations whose sums differ in sign.

Here is what the three fields do, stated in terms of the null space that Theorem 1 showed to be the difficulty. $\sigma$ answers the first two rows of the decision table by itself. $F$ bounds the null space from below: $V_F \subseteq \operatorname{null}(H)$ when $H \succeq 0$, which is Proposition 1. $D$ bounds it from above: $V_D \cap \operatorname{null}(H) = \{0\}$. Each is only a bound on its own. A covering tag turns the pair into an equality.

:::{admonition} Corollary 1 (a covering tag pins the null space)
:class: important

Let $H \succeq 0$ with $H \models t$, and let $t$ be covering. Then $\operatorname{null}(H) = V_F$.

:::

:::{admonition} Proof
:class: dropdown

$V_F \subseteq \operatorname{null}(H)$ is the first part of Proposition 1. For the reverse, let $v \in \operatorname{null}(H)$. Since $F \cup D = \{1, \dots, n\}$, the coordinates of $F$ and $D$ together are all of them, so $v = f + d$ with $f \in V_F$ and $d \in V_D$. Then $f \in \operatorname{null}(H)$, hence $d = v - f \in \operatorname{null}(H)$, so $d \in V_D \cap \operatorname{null}(H) = \{0\}$ by the second part of Proposition 1. Therefore $v = f \in V_F$.

:::

:::{admonition} Example (the ridge penalty of a GLM)
:class: note

Take the model of the problem setting: two leaves, `coef` and `intercept`, and a ridge penalty of strength $\lambda > 0$ that does not reach the intercept, so that

$$\nabla^2 p_\lambda = \begin{pmatrix} \lambda I & 0 \\ 0 & 0 \end{pmatrix} .$$

It satisfies the tag $t = ({\succeq}\,0, \ F, \ D)$ with $F = \{\texttt{intercept}\}$ and $D = \{\texttt{coef}\}$: the matrix is positive semidefinite, the intercept block is $0$, and the coefficient block is $\lambda I \succ 0$. The two sets cover the tree, so $t$ is covering and Corollary 1 applies:

$$\operatorname{null}(\nabla^2 p_\lambda) = V_{\{\texttt{intercept}\}} = \operatorname{span}\{e_{p+1}\} ,$$

which is the null space read off by inspection in the problem setting, now obtained from the tag alone.

:::

The structure of the matrix — `Full`, `BlockDiagonal`, `Diagonal` — is deliberately not part of $t$. It answers the last two rows of the decision table on its own and never interacts with the sign or the null space, so it is treated separately, at the end of this note.

:::{admonition} Definition (sound rule, best rule)
:class: note

A rule $R : \mathbb{T} \times \mathbb{T} \to \Sigma$ is *sound* if

$$H_1 + H_2 \models R(t_1, t_2) \qquad \text{whenever } H_1 \models t_1 \text{ and } H_2 \models t_2 .$$

The *best rule* is

$$R^*(t_1, t_2) = \max_{\sqsubseteq} \, \big\{ s \in \Sigma : H_1 + H_2 \models s \text{ for all } H_1 \models t_1, \ H_2 \models t_2 \big\} .$$

:::

$R^*$ is well defined: the set is non-empty, since `none` holds of everything, and $\Sigma$ is a finite chain. It is sound by construction, and every sound rule $R$ satisfies $R(t_1, t_2) \sqsubseteq R^*(t_1, t_2)$ pointwise, so it is the strongest sound rule there is. What remains is to compute it, and to check that the rule the package implements agrees.

Three statements do that, in order:

- **Proposition 3**: exactly which tags are realizable.
- **Theorem 2**: a closed form for $R^*$.
- **Theorem 3**: the implemented rule equals $R^*$ on realizable tags.

## Which tags are realizable

A tag is realizable when $H \models t$ for some $H \in \operatorname{Sym}(N)$. Two conditions settle which ones are.

:::{admonition} Proposition 3 (realizability)
:class: important

A tag $t = (\sigma, F, D)$ is realizable if and only if

$$\text{(i)} \quad F \cap D = \varnothing , \qquad\qquad \text{(ii)} \quad \sigma = {\succ}\,0 \implies F = \varnothing .$$

:::

:::{admonition} Proof
:class: dropdown

*Necessity.* Let $H \models t$ and let $i \in F$. The leaves are non-empty, so pick a coordinate $j \in I_i$. It is a coordinate of $F$, so $H[F] = 0$ gives

$$H_{jj} = 0 .$$

Each of the two conditions now follows from a different positivity assumption contradicting this.

Suppose $i \in D$ as well. Then $j$ is a coordinate of $D$, so $e_j \in V_D$ and the restriction identity from the notation section applies:

$$H_{jj} = e_j[D]^\top \, H[D] \, e_j[D] > 0 ,$$

the inequality because $e_j[D]$ is a non-zero vector and $H[D] \succ 0$. This contradicts $H_{jj} = 0$, so no leaf lies in both sets, that is $F \cap D = \varnothing$.

Suppose instead $\sigma = {\succ}\,0$, so $H \succ 0$. Then $H_{jj} = e_j^\top H e_j > 0$. Hence $F$ contains no leaf at all, that is $F = \varnothing$.

*Sufficiency.* Let $t = (\sigma, F, D)$ satisfy (i) and (ii). We exhibit an $H$ with $H \models t$.

If $\sigma = {\succ}\,0$, then (ii) gives $F = \varnothing$, and $H = I$ works: it is positive definite, the requirement on $F$ is vacuous, and $I[D] = I \succ 0$.

Otherwise $\sigma$ is ${\succeq}\,0$ or `none`. Take $P$, the orthogonal projector onto $V_F^{\perp}$: it is symmetric, satisfies $P = P^2$ and $\operatorname{null}(P) = V_F$, and is positive semidefinite, so $P \models \sigma$, since ${\succeq}\,0$ implies `none`.

If $F = \varnothing$, then $V_F = \{0\}$ and $P = I$, and the requirement on $F$ is vacuous, $P[F]$ being the empty matrix.

If $F \neq \varnothing$, pick $j \in I_F$. The basis vector $e_j$ lies in $V_F = \operatorname{null}(P)$, so $P e_j = 0$ and the $j$-th column of $P$ vanishes. Taking $k, j \in I_F$ gives $P_{kj} = 0$ throughout, that is $P[F] = 0$.

There remains $P[D] \succ 0$. Let $v \in V_D$ be non-zero. Using the restriction identity, and then $P = P^\top = P^2$,

$$v[D]^\top P[D] \, v[D] = v^\top P v = v^\top P^\top P v = \lVert P v \rVert^2 ,$$

which vanishes only when $v \in \operatorname{null}(P) = V_F$. But $v \in V_D$, and condition (i) gives $V_F \cap V_D = \{0\}$, so $v \in V_F$ would force $v = 0$, contrary to assumption. Hence $\lVert P v \rVert^2 > 0$.

As $v$ ranges over the non-zero vectors of $V_D$, its restriction $v[D]$ ranges over all non-zero vectors of $\mathbb{R}^{d(D)}$, so what has just been shown is exactly $P[D] \succ 0$.

:::

Two consequences. In a realizable covering tag, $F$ and $D$ partition the leaves: condition (i) makes them disjoint, and covering means their union is everything.

And nothing constrains $D$ against $\sigma$. Take $t = ({\succeq}\,0, \varnothing, \{1, \dots, n\})$. It is realizable, since $I \models t$. It is also covering, so Corollary 1 applies to every $H \models t$ and gives $\operatorname{null}(H) = V_\varnothing = \{0\}$; together with $H \succeq 0$ that makes $H$ positive definite. So $t$ is satisfied by exactly the positive definite matrices while claiming only ${\succeq}\,0$. That is under-claiming, not inconsistency — a tag states what its declarer is prepared to guarantee, not the strongest truth available.

## Normalizing a tag

Two tags with the same satisfying set are interchangeable, because $R^*$ is defined by quantifying over the matrices that satisfy each argument and so cannot tell them apart. Two kinds of declaration are weaker than they need to be, and each can be strengthened without changing the matrices it describes.

:::{admonition} Definition (normalization)
:class: note

For a realizable $t = (\sigma, F, D)$ put

$$\mathrm{nf}(t) = \begin{cases}
({\succ}\,0, \ \varnothing, \ \{1, \dots, n\}) & D = \{1, \dots, n\}, \\[2pt]
({\succeq}\,0, \ \{1, \dots, n\}, \ \varnothing) & F = \{1, \dots, n\}, \\[2pt]
t & \text{otherwise.}
\end{cases}$$

:::

:::{admonition} Proposition 4 (normalization preserves the satisfying set)
:class: important

$H \models t \iff H \models \mathrm{nf}(t)$ for every realizable $t$ and every $H \in \operatorname{Sym}(N)$. Consequently $R^*(t_1, t_2) = R^*\big(\mathrm{nf}(t_1), \mathrm{nf}(t_2)\big)$.

:::

:::{admonition} Proof
:class: dropdown

In each of the first two branches we show that $H \models t$ and $H \models \mathrm{nf}(t)$ are equivalent to one and the same explicit condition on $H$, hence to each other.

*Branch $D = \{1, \dots, n\}$*, where $\mathrm{nf}(t) = ({\succ}\,0, \varnothing, \{1, \dots, n\})$. Condition (i) gives $F \cap D = \varnothing$, so $F = \varnothing$ and the requirement $H[F] = 0$ is vacuous for $t$; it is vacuous for $\mathrm{nf}(t)$ as well. For $t$,

$$H \models t \iff H \models \sigma \ \text{ and } \ H[D] = H \succ 0 \iff H \succ 0 ,$$

the second equivalence because $H \succ 0$ implies $H \models \sigma$ for every $\sigma \in \Sigma$, ${\succ}\,0$ being the greatest element. For $\mathrm{nf}(t)$ the two requirements are $H \succ 0$ and $H[\{1, \dots, n\}] = H \succ 0$, which is the same condition twice. So both sides are $H \succ 0$.

*Branch $F = \{1, \dots, n\}$*, where $\mathrm{nf}(t) = ({\succeq}\,0, \{1, \dots, n\}, \varnothing)$. Condition (i) gives $D = \varnothing$, so the requirement on $D$ is vacuous on both sides, and (ii) excludes $\sigma = {\succ}\,0$, leaving $\sigma \in \{\text{none}, {\succeq}\,0\}$. For $t$,

$$H \models t \iff H \models \sigma \ \text{ and } \ H[F] = H = 0 \iff H = 0 ,$$

the second equivalence because $0 \models \text{none}$ and $0 \models {\succeq}\,0$. For $\mathrm{nf}(t)$ the requirements are $H \succeq 0$ and $H = 0$, again $H = 0$. So both sides are $H = 0$.

The third branch sets $\mathrm{nf}(t) = t$ and needs no argument. The statement about $R^*$ then follows because $R^*$ depends on its arguments only through the sets $\{H : H \models t_i\}$, which the first two branches leave unchanged.

:::

Both rewrites replace an under-claiming declaration with a stronger one that the tag already implies, so neither adds information the tag did not have. A model may declare `none` and get a positive definite tag back from the normalizer, because of its own $D$.

## Combining two tags: the rule, and why it is sound

$R^*$ quantifies over every pair of matrices satisfying the two tags, which is not something a rule can evaluate. The quantifier is removed in two steps: first the sum's sign is expressed through null spaces, then the achievable null spaces are read off the tag.

The first step is the identity already used in the problem setting. For $A, B \succeq 0$,

$$A + B \succ 0 \iff \operatorname{null}(A) \cap \operatorname{null}(B) = \{0\} ,$$

since $\operatorname{null}(A + B) = \operatorname{null}(A) \cap \operatorname{null}(B)$ and a positive semidefinite matrix is definite exactly when its null space is trivial. The second step is the following.

:::{admonition} Proposition 5 (achievable null spaces)
:class: important

Let $t = (\sigma, F, D)$ be realizable. If $\sigma = {\succeq}\,0$ then

$$\big\{ \operatorname{null}(H) : H \models t \big\} = \big\{ L \ \text{a subspace of } \mathbb{R}^N \ : \ V_F \subseteq L, \ L \cap V_D = \{0\} \big\} ,$$

and if $\sigma = {\succ}\,0$ then $\{ \operatorname{null}(H) : H \models t \} = \{\{0\}\}$.

:::

:::{admonition} Proof
:class: dropdown

*Case $\sigma = {\succeq}\,0$.* Write $\mathcal{N}$ for the set on the left and $\mathcal{L}$ for the family on the right; we prove both inclusions.

*($\mathcal{N} \subseteq \mathcal{L}$).* Let $H \models t$, so $H \succeq 0$, $H[F] = 0$ and $H[D] \succ 0$. Proposition 1 gives $V_F \subseteq \operatorname{null}(H)$ and $V_D \cap \operatorname{null}(H) = \{0\}$, which are the two defining conditions of $\mathcal{L}$. So $\operatorname{null}(H) \in \mathcal{L}$.

*($\mathcal{L} \subseteq \mathcal{N}$).* Let $L \in \mathcal{L}$ and let $P$ be the orthogonal projector onto $L^{\perp}$. Being an orthogonal projector, $P$ is symmetric with $P = P^2$ and $v^\top P v = \lVert P v \rVert^2 \ge 0$, so $P \succeq 0$, and $\operatorname{null}(P) = L$. For $j \in I_F$ we have $e_j \in V_F \subseteq L = \operatorname{null}(P)$, so $P e_j = 0$, the $j$-th column vanishes, and taking $k, j \in I_F$ gives $P[F] = 0$. For non-zero $v \in V_D$ the restriction identity gives $v[D]^\top P[D] \, v[D] = \lVert P v \rVert^2$, non-zero because $v \notin L$; so $P[D] \succ 0$. Hence $P \models t$ with $\operatorname{null}(P) = L$, that is $L \in \mathcal{N}$. This is the construction of Proposition 3 with $L$ in place of $V_F$.

*Case $\sigma = {\succ}\,0$.* Every $H \models t$ is positive definite, hence non-singular, so the only achievable null space is $\{0\}$.

:::

The two cases must be kept apart. For $\sigma = {\succ}\,0$ the subspace family on the right can be strictly larger than what is achievable: with two leaves, $t = ({\succ}\,0, \varnothing, \{1\})$ admits $L = V_{\{2\}}$ into the family, while every $H \models t$ is non-singular.

### Optimal closed form combination rule

:::{admonition} Theorem 2 (the best rule)
:class: important

Let $t_1, t_2$ be realizable and normalized. Say the pair is *linked* when

$$\text{$t_1$ is covering and } F_1 \subseteq D_2, \qquad\text{or}\qquad \text{$t_2$ is covering and } F_2 \subseteq D_1 .$$

Then

$$R^*(t_1, t_2) = \begin{cases}
\text{none} & \sigma_1 = \text{none or } \sigma_2 = \text{none}, \\[2pt]
{\succ}\,0 & \text{both signed, and } \sigma_1 = {\succ}\,0 \text{ or } \sigma_2 = {\succ}\,0 \text{ or the pair is linked}, \\[2pt]
{\succeq}\,0 & \text{otherwise.}
\end{cases}$$

:::

:::{admonition} Proof
:class: dropdown

*The `none` case.* Say $\sigma_1 = \text{none}$. Normalization gives $D_1 \neq \{1, \dots, n\}$ and $F_1 \neq \{1, \dots, n\}$. Fix any $H_2 \models t_2$, take $P \models t_1$ from Proposition 3, and perturb it.

If $t_1$ is not covering, pick a leaf $i \notin F_1 \cup D_1$ and $j \in I_i$, and put $H_1 = P - c\, e_j e_j^\top$. Neither $H_1[F_1]$ nor $H_1[D_1]$ sees the perturbation, since $j$ belongs to neither, so $H_1 \models t_1$ for every $c$, while $e_j^\top (H_1 + H_2) e_j \to -\infty$.

If $t_1$ is covering, then $F_1$ and $D_1$ are both non-empty, by normalization again, so pick $j \in I_{F_1}$ and $k \in I_{D_1}$ and put $H_1 = P + c\,(e_j e_k^\top + e_k e_j^\top)$. The perturbation touches only the entries $(j,k)$ and $(k,j)$, which lie in neither $F_1 \times F_1$ nor $D_1 \times D_1$, so again $H_1 \models t_1$ for every $c$. The block of $H_1 + H_2$ on $\{j, k\}$ has determinant $(H_2)_{jj} (H_1 + H_2)_{kk} - (c + x)^2$ for a constant $x$, which tends to $-\infty$.

Either way $H_1 + H_2$ is indefinite for large $c$, so no sign holds of every pair and $R^* = \text{none}$.

*Both signed.* Every $H_i \models t_i$ is positive semidefinite, so every sum is, and $R^* \sqsupseteq {\succeq}\,0$. By the null space identity, $R^* = {\succ}\,0$ exactly when $\operatorname{null}(H_1) \cap \operatorname{null}(H_2) = \{0\}$ for all admissible pairs, which by Proposition 5 is a condition on the families $\mathcal{L}_i$.

If $\sigma_1 = {\succ}\,0$ then $\operatorname{null}(H_1) = \{0\}$ always and the intersection is trivial; likewise for $\sigma_2$. So assume both signs are ${\succeq}\,0$.

*Linked implies definite.* Say $t_1$ is covering with $F_1 \subseteq D_2$. Covering and disjointness give $V_{F_1} \oplus V_{D_1} = \mathbb{R}^N$, so for $L_1 \in \mathcal{L}_1$ and $w \in L_1$, writing $w = f + d$ with $f \in V_{F_1} \subseteq L_1$ leaves $d = w - f \in L_1 \cap V_{D_1} = \{0\}$; hence $L_1 = V_{F_1}$ is the only member. Then for any $L_2 \in \mathcal{L}_2$,

$$L_1 \cap L_2 = V_{F_1} \cap L_2 \subseteq V_{D_2} \cap L_2 = \{0\} ,$$

using $F_1 \subseteq D_2$. So every admissible pair meets trivially.

*Not linked implies not definite.* Suppose neither disjunct holds; we exhibit $L_1, L_2$ meeting non-trivially.

If neither tag is covering, then $V_{F_1 \cup D_1}$ and $V_{F_2 \cup D_2}$ are proper subspaces of $\mathbb{R}^N$. A vector space is never the union of two proper subspaces: given $a \notin A$ and $b \notin B$ with $a \in B$, $b \in A$, the vector $a + b$ lies in neither. So pick $v$ outside both and set $L_i = V_{F_i} + \operatorname{span}\{v\}$. Each is admissible: if $f + \alpha v \in V_{D_i}$ with $f \in V_{F_i}$, then $\alpha v \in V_{F_i \cup D_i}$, forcing $\alpha = 0$, and then $f \in V_{F_i} \cap V_{D_i} = \{0\}$. And $v$ lies in both.

Otherwise some tag is covering, say $t_1$, and $F_1 \not\subseteq D_2$. Pick a leaf $i \in F_1 \setminus D_2$ and $j \in I_i$, and put $v = e_j \in V_{F_1} = L_1$. If $i \in F_2$, take $L_2 = V_{F_2}$, which is admissible and contains $v$. If $i \notin F_2 \cup D_2$, take $L_2 = V_{F_2} + \operatorname{span}\{v\}$, admissible by the computation just given since $v \notin V_{F_2 \cup D_2}$. Either way $v \in L_1 \cap L_2$ is non-zero.

So $R^* = {\succ}\,0$ exactly when the pair is linked, and ${\succeq}\,0$ otherwise.

:::

The clause on $\sigma_i = {\succ}\,0$ can be absorbed. A tag with $\sigma = {\succ}\,0$ has satisfying set $\{H : H \succ 0\}$, since realizability forces $F = \varnothing$ and $H \succ 0$ implies $H[D] \succ 0$; so it has the same satisfying set as $({\succ}\,0, \varnothing, \{1, \dots, n\})$, which is covering with $F = \varnothing \subseteq D_2$. Extending $\mathrm{nf}$ with that rewrite would leave *linked* as the only condition to test.

## Is this the right tagging system?

Theorem 2 says the rule is the best one *for this tag*, and says nothing about the tag. A description that recorded more could support a finer rule, so it is worth asking whether ours leaves anything behind. It does not, but for two different reasons on the two sides of the sum.

:::{admonition} Definition (refinement)
:class: note

A *tagging system* is a map $\delta : \operatorname{Sym}(N) \to \Delta$ assigning to each matrix a description drawn from some set $\Delta$. A system $\delta' : \operatorname{Sym}(N) \to \Delta'$ *refines* $\delta$ if there is a map $g : \Delta' \to \Delta$ with

$$\delta = g \circ \delta' ,$$

that is, if $\delta'$ determines $\delta$ and so records at least as much about $H$.

:::

If $\delta'$ refines $\delta$ then $R^*_{\delta'} \sqsupseteq R^*_{\delta}$ pointwise, since a finer description cuts down the set of consistent pairs the maximum ranges over. The candidate refinement is to record, instead of one flat set and one definite set, the whole families

$$\mathcal{F}(H) = \{F : H[F] = 0\}, \qquad \mathcal{D}(H) = \{D : H[D] \succ 0\} ,$$

which is what $\tau$ does. A single set can be weaker than a family, because $\mathcal{D}(H)$ need not be closed under union: a matrix can be definite on two sets of leaves and singular on the union, in which case one set cannot carry both facts.

### For a penalty the single set is already the family

Assumption 2 rules that case out. A penalty is a sum of terms, one per leaf, so its Hessian has no coupling between leaves.

:::{admonition} Proposition 6 (no coupling makes the definite family union-closed)
:class: important

Let $H$ be block diagonal with respect to the leaf partition. Then $H[S] \succ 0$ if and only if $H[\{i\}] \succ 0$ for every $i \in S$. Consequently $\mathcal{D}(H)$ is closed under union and has a greatest element, the set of all leaves whose own block is definite.

:::

:::{admonition} Proof
:class: dropdown

No coupling between leaves means $H[S] = \bigoplus_{i \in S} H[\{i\}]$ for every set $S$ of leaves. A direct sum of symmetric matrices is positive definite exactly when every summand is, which is the stated equivalence. If $H[S_1] \succ 0$ and $H[S_2] \succ 0$ then every leaf of $S_1 \cup S_2$ has a definite block, so $H[S_1 \cup S_2] \succ 0$; and the union of all definite leaves is itself definite, hence greatest.

:::

So for any term satisfying Assumption 2, the single set $D$ can be taken to be the greatest element of $\mathcal{D}(H)$, and it then carries exactly as much as the family. This holds for more than the regularizers nemos has today: any penalty that is a strength-weighted sum over leaves has the property, whatever its shape, so the refinement would buy nothing for it.

### For a model the refinement is possible, and is what we are avoiding

The loss Hessian does couple leaves, $\nabla^2 \ell = \tilde{X}^\top W \tilde{X}$ being dense in general, so its definite family need not be union-closed and a single set can genuinely lose.

Concretely, take a design built from two blocks, $X = \begin{pmatrix} X_1 & X_2 \end{pmatrix}$, each a `BSplineEval` basis, so that the parameters have leaves $\texttt{coef}_1$ and $\texttt{coef}_2$ and

$$\nabla^2 \ell [\{\texttt{coef}_1\}] = X_1^\top W X_1, \qquad \nabla^2 \ell [\{\texttt{coef}_2\}] = X_2^\top W X_2 ,$$

both definite, since each basis is of full column rank. Their union is not. The features of each block sum to one across columns at every sample, $X_1 \mathbf{1} = X_2 \mathbf{1} = \mathbf{1}$, so the coefficient vector $v = (\mathbf{1}, -\mathbf{1})$, ones on the first leaf and minus ones on the second, has

$$X v = X_1 \mathbf{1} - X_2 \mathbf{1} = 0, \qquad v^\top \nabla^2 \ell \, v = \lVert W^{1/2} X v \rVert^2 = 0 .$$

So $\mathcal{D}$ contains $\{\texttt{coef}_1\}$ and $\{\texttt{coef}_2\}$ and not $\{\texttt{coef}_1, \texttt{coef}_2\}$: exactly the failure of union-closure that Proposition 6 rules out for penalties, arising here from an ordinary additive model.

But the loss cannot exploit a richer tag, and the obstruction is not the tag's shape. Deciding whether $\tilde{X}^\top W \tilde{X}$ is definite on a group of columns is deciding whether those columns are independent under the weights, and every route to that answer — a Cholesky attempt, a rank computation by QR or SVD — costs $O(d^3)$, the same order as the Newton solve the tag exists to route. A design-aware tag would require a factorization in order to choose a factorization.

Nor could the information arrive from elsewhere. A model receives $X$ as an array, not the basis that produced it, so the fact that a design sums to one across columns is known to `BSplineEval` and unavailable to the `GLM`.

What remains derivable before looking at any array is exactly what the class-level tag carries: the Hessian is positive semidefinite, from convexity of the link, and it is definite on the intercept, because $\mathbf{1}^\top W \mathbf{1} = \sum_t w_t(\beta) > 0$ is a sum of non-negative terms and needs no factorization. Every other entry of $\mathcal{D}$ is a rank question about the data.

The refinement is sound mathematics, and this note does not adopt it. For penalties, Proposition 6 shows the single set already carries everything the family carries, so a finer tag would say no more. For models, the extra information can only be obtained by computing a factorization, which is $O(d^3)$, the same order as the Newton solve the tag selects a method for.

### The ridge-penalized GLM, end to end

Two leaves, `coef` and `intercept`. The loss and the penalty declare

$$t_\ell = \big({\succeq}\,0, \ \varnothing, \ \{\texttt{intercept}\}\big), \qquad
t_p = \big({\succeq}\,0, \ \{\texttt{intercept}\}, \ \{\texttt{coef}\}\big) .$$

For $t_\ell$: the sign is Assumption 4 applied to a convexity-preserving link, $F$ is empty because the loss is zero on no leaf, and $\texttt{intercept} \in D$ by the sum of weights. For $t_p$: the sign holds because $\lambda \ge 0$, $\texttt{intercept} \in F$ because the penalty does not reach it, and $\texttt{coef} \in D$ because every $\lambda_j$ on that leaf is strictly positive.

Both tags are signed, so Theorem 2 asks whether the pair is linked. It is: $t_p$ is covering, since $\{\texttt{intercept}\} \cup \{\texttt{coef}\}$ is every leaf, and

$$F_p = \{\texttt{intercept}\} \subseteq \{\texttt{intercept}\} = D_\ell .$$

Hence $R^*(t_\ell, t_p) = {\succ}\,0$: the penalized Hessian is positive definite, and Cholesky is safe. Neither term is definite on its own, and this is the conclusion the opening example reached by hand.

Two of the hypotheses are visible in that computation. A zero strength on any coefficient would remove $\texttt{coef}$ from $D_p$, leaving $t_p$ non-covering and the verdict ${\succeq}\,0$. A link that does not keep the likelihood convex would leave $\sigma_\ell = \text{none}$, and the verdict would be `none` whatever the penalty.

## Structure composes on its own

Say $H$ is *block diagonal with respect to a partition $\mathcal{P}$ of the coordinates* when $H_{jk} = 0$ whenever $j$ and $k$ lie in different blocks of $\mathcal{P}$. If $H_1$ is block diagonal with respect to $\mathcal{P}_1$ and $H_2$ with respect to $\mathcal{P}_2$, then $H_1 + H_2$ is block diagonal with respect to the join $\mathcal{P}_1 \vee \mathcal{P}_2$, whose blocks are the connected components of the union of the two patterns. Nothing finer holds in general.

Two cases are easy, because the diagonal partition refines every other: diagonal plus diagonal is diagonal, and diagonal plus block diagonal is block diagonal with the same partition. The third is not. With $\mathcal{P}_1 = \{\{1,2\},\{3,4\}\}$ and $\mathcal{P}_2 = \{\{1\},\{2,3\},\{4\}\}$ the join is the single block $\{1,2,3,4\}$: the sum is tridiagonal and has no non-trivial block structure at all.

In nemos the case that arises is narrower. Block structure only ever comes from vmapping over a batch axis, so the partition is that axis by construction and is the same for the model term and the penalty term: $\mathcal{P}_1 = \mathcal{P}_2$, and the join is that partition again. Under that hypothesis the ordering `Diagonal` $<$ `BlockDiagonal` $<$ `Full`, combined by taking the larger, is correct. Without it, two block diagonal terms can sum to something the ordering would call `BlockDiagonal` and that is in fact `Full`.

## Where each piece lives

Everything in the table is in `src/nemos/_hess.py` unless another module is named.

| in this note | in the code |
| --- | --- |
| $\sigma \in \Sigma$ | `MatrixProperty`, where $\text{none}$ is spelled `SYMMETRIC` |
| $F$, $D$ | `HessianTag.flat_on`, `HessianTag.definite_on`: trees shaped like the parameters holding one boolean per leaf, so a set of leaves is a mask and the set operations of the combination rule are `tree_map` |
| the per-leaf verdict the two sets are read off | `LeafClaim.FLAT`, `LeafClaim.DEFINITE`, `LeafClaim.UNCLAIMED`, one per leaf, split into the two masks by `mask_of_claim` |
| $\sigma_\ell$ | `BaseRegressor._resolve_hess_property`, overridden by `GLM` to look the inverse link up in the observation model's convexity-preserving list |
| $D_\ell$ | `BaseRegressor._hess_leaf_claims`, where `GLM` claims the intercept and the classifiers claim nothing |
| $t_p$ | `Regularizer._resolve_hess_tag`, with `_leaf_claim` turning the strength on one leaf into that leaf's claim |
| the normalizing map | `normalize` |
| $R^*$ | `combine_hessian_tags`, which is `combine_property` applied over `combine_definite_on` |
| the structure ordering | `MatrixStructure`, an `IntEnum` whose value is how general the structure is, so "take the larger" is `max` |
| the decision the tag is for | `Newton.init_state` in `nemos/solvers/_newton.py`: `POSITIVE_DEFINITE` selects `lx.Cholesky` and tags the operator semidefinite, anything weaker selects `lx.AutoLinearSolver(well_posed=False)` |

The tag is built when the solver is set up, in `BaseRegressor._instantiate_solver`, against the parameters actually being fitted. A parameter held fixed is `None` in that tree, and every `tree_map` above drops it together with whatever was claimed about it. A claim about a pinned leaf therefore disappears with the leaf; it does not become an unclaimed leaf.

Two places where the code carries more than the derivation above needs. `MatrixProperty` also has `NEGATIVE_SEMI_DEFINITE` and `NEGATIVE_DEFINITE`, handled by the $-H$ symmetry the definition of $\Sigma$ already invokes; no loss or penalty in the package declares one. And a leaf carries `UNCLAIMED` rather than being absent from both masks, which is the same statement as lying in neither $F$ nor $D$, with the enum making it impossible to claim a leaf both flat and definite.

Assumption 3 is enforced rather than assumed: `Regularizer._validate_strength` rejects a negative strength leaf by leaf, so `_leaf_claim` only ever compares non-negative entries.
