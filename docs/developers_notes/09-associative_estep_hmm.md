# Associative Scan For The HMM E-Step

This note will clarify the math behind the associative scan implementation of the E-step of the HMM.

## Notation
| object | meaning |
| --- | --- |
| $K$, $T$ | number of states, number of time bins |
| $b_t \in \mathbb{R}_+^K$ | $b_t[i] = p(y_t \mid z_t = i)$ |
| $A \in \mathbb{R}_+^{K \times K}$ | $A[j,i] = p(z_t = i \mid z_{t-1} = j)$, rows sum to $1$ |
| $\pi \in \mathbb{R}_+^K$ | $\pi[i] = p(z_t = i)$ at a session start |
| $S$ | set of session-start indices, always $0 \in S$ |
| $s(t)$ | start of the session containing $t$, $s(t) = \max\{u \in S : u \leq t\}$ |

## Scan Map

Let $\oplus$ be an associative operator on a set $\mathcal{M}$. For $x=(x_0,\dots,x_{T-1})\in \mathcal{M}^T$, let $X_{a:b} = x_a \oplus \dots \oplus x_b$. The *scan map* is

$$
\mathcal{S}: \mathcal{M}^T \rightarrow \mathcal{M}^T, \;\;\; \mathcal{S}(x) = (X_{0:0}, \dots X_{0:T-1})
$$

Since $\oplus$ is associative, the scan map can be computed efficiently via `jax.lax.associative_scan`.

## The forward pass associativity

The forward step computes two quantities:

$$
\hat{\alpha}_t[i] \;=\; p\bigl(z_t = i \mid y_{s(t):t}\bigr), \qquad c_t \;=\; p\bigl(y_t \mid y_{s(t):t-1}\bigr),
$$

These quantities can be computed recursively as follows:

$$
\tilde{\alpha}_t = \begin{cases}
\pi \odot b_t, & t \in S, \\
(\hat{\alpha}_{t-1} A) \odot b_t, & t \notin S,
\end{cases}
$$

$$
c_t = \tilde{\alpha}_t \mathbf{1}, \qquad \hat{\alpha}_t = \tilde{\alpha}_t / c_t .
$$

Now, we can calculate these quantities by associative scan if we define:

$$
F_{u:t}[j,i]=\begin{cases}
p(y_{u:t},\, z_t=i | z_{u-1} = j), & u \notin S, \\
p(y_{u:t},\, z_t=i), & u \in S.
\end{cases}
$$

We can notice that $F_{u:v}$ has the following properties:

:::{admonition} Proposition 1 (one-step update)

$$
F_{t:t} = \begin{cases}
    \mathbf{1} (\pi \odot b_t)^\top & t \in S, \\
    A \cdot \text{diag} (b_t) & t \notin S.
\end{cases}
$$

For all $t=0,\dots,T-1$.
:::

:::{admonition} Proof
:class: dropdown

Follows directly from the HMM equations.
:::



:::{admonition} Proposition 2 (matrix product)

For any $0 \leq u \leq v < t \leq T$, $F_{u:t} = F_{u:v} \cdot F_{v+1:t}$.

:::

:::{admonition} Proof
:class: dropdown

Case $u \notin S$.

$$
\begin{aligned}
F_{u:t}[j, i] &= p(y_{u:t}, z_t = i \mid  z_{u-1} = j) \\
&= \sum_k p(y_{u:t}, z_t = i, z_v = k \mid z_{u-1} = j) \\
&= \sum_k p(y_{v+1:t}, z_t = i \mid y_{u:v}, z_v = k, z_{u-1} = j) p(y_{u:v}, z_v = k \mid z_{u-1} = j) \\
&= \sum_k F_{u:v}[j,k]   p(y_{v+1:t}, z_t = i \mid y_{u:v}, z_v = k, z_{u-1} = j)\\
&\overset{(\star)}{=} \sum_k F_{u:v}[j,k]   p(y_{v+1:t}, z_t = i \mid z_v = k),
\end{aligned}
$$


Where $(\star)$ holds for the conditional independence implied by the HMM graphical model.

If $v+1 \notin S$,

$$
p(y_{v+1:t}, z_t = i \mid z_v = k) = F_{v+1:t}[k,i].
$$

If $v+1 \in S$, the chain restarts after $v$, which implies that $y_{v+1:t}$ and $z_t$ are independent from $z_v$. From this we have,

$$
p(y_{v+1:t}, z_t = i) = F_{v+1:t}[k, i],
$$

where the last is by definition of $F_{u:v}$ for $u \in S$. Therefore, for all $v+1$

$$
F_{u:t}[j, i] = \sum_k F_{u:v}[j,k] \cdot F_{v+1:t}[k, i].
$$

Case $u \in S$.

Same computation but without the conditioning on $z_{u-1}$.

:::

This last proposition implies that $F_{0:t} = \prod_{r=0}^t F_{r:r}$ for all $t=0,\dots,T-1$, a matrix product, which is associative.

:::{admonition} Proposition ($\hat{\alpha}_t$ and $c$ from $F_{0:t}$)
:class: note

We have that

$$
\hat{\alpha}_t = \frac{F_{0:t}[j,\cdot]}{F_{0:t}[j,\cdot] \mathbf{1}}
$$

and

$$
\prod_{r\leq t} c_r = F_{0:t}[j,\cdot] \mathbf{1}
$$

for all $j=0,\dots,K-1$.

:::


:::{admonition} Proof
:class: dropdown

$\hat{\alpha}_t = p(z_t = \cdot \mid y_{s(t):t})$ by definition.

Rewriting $F_{s(t):t}$ via the Bayes rule we have:

$$
\begin{aligned}
F_{s(t):t}[j,\cdot] &= p(y_{s(t):t}, z_t = \cdot) \\
&= p(y_{s(t):t}) \frac{p(y_{s(t):t}, z_t = \cdot)}{p(y_{s(t):t})}\\
&= p(y_{s(t):t}) p(z_t = \cdot | y_{s(t):t})\\
&= p(y_{s(t):t}) \hat{\alpha}_t[\cdot],
\end{aligned}
$$

since $s(t) \in S$. Note how $F_{s(t):t}[j,\cdot]$ is constant for all $j$, i.e. all the rows of $F_{s(t):t}$ are equal.

By proposition 2,

$$
\begin{aligned}
F_{0:t} & = F_{0:s(t)-1} \,F_{s(t):t} \\
&= \sum_k F_{0:s(t)-1}[\cdot \,, k] p(y_{s(t):t})  \hat{\alpha}_t[\cdot]\\
&= \big(\sum_k F_{0:s(t)-1}[\cdot\, , k]\big) \, p(y_{s(t):t})  \hat{\alpha}_t[\cdot] \\
&= \big(\sum_k p(y_{0:s(t)-1}, z_{s(t)-1}=k) \big) \, p(y_{s(t):t}) \hat{\alpha}_t[\cdot]\\
&= p(y_{0:s(t)-1})  p(y_{s(t):t}) \hat{\alpha}_t[\cdot] \\
&= p(y_{0:t})\hat{\alpha}_t[\cdot],
\end{aligned}
$$

where the last one holds for the independence of the sessions. Substituting $F_{0:t}[j, \cdot] \mathbf{1}=p(y_{0:t})$, and solving for $\hat{\alpha}_t$ proves the first claim.

For the second claim, we can notice that within one session, by definition of $c_t$,

$$
\prod_{r=s(t)}^t c_r = p(y_{s(t)}) \prod_{r=s(t)+1}^t p(y_r \mid y_{s(t):r-1})= p(y_{s(t):t})
$$

Therefore, for independence across sessions, $\prod_{r\leq t} c_r = p(y_{0:t}) = F_{0:t}[j, \cdot] \mathbf{1}$.
:::

An associative scan over $F$s is possible, and would recover the forward step parameters, however, this approach would not work numerically. In particular, $F_{u:v}$ are joint probabilities over $v-u+1$ observations, and therefore decay geometrically with the length of the sequence, and the running products in the scan would rapidly underflow. Another way to see this is considering that $F_{0:t} = \prod_{r=0}^{t} F_{r:r}$: each time we take a product we multiply probabilities in $[0,1]$, creating an exponential decay. In the next section we introduce an alternative parametrization that prevents that from happening while still preserving the associativity of the operations involved in the scan.

## Stable Parametrization

The idea is to decompose $F_{u:t}$ in its row sums and the rest,

$$
l_{u:t} = F_{u:t} \mathbf{1}, \qquad L_{u:t} = \text{diag}(l_{u:t})^{-1} F_{u:t}.
$$

The first term is $l_{u:t}[j] = p(y_{u:t} \mid z_{u-1} = j)$, which is strictly positive for every observation model in nemos — Poisson, Bernoulli, Categorical, Gamma and Gaussian all have strictly positive densities, so $b_t > 0$ — and can therefore be safely inverted, as in the formula above. Note that positivity is all that is needed: $l_{u:t} \leq 1$ holds for the discrete emissions but not for the continuous ones, whose densities can exceed $1$. The second term is

$$
\begin{aligned}
L_{u:t}[j,i] &=\; \frac{F_{u:t}[j,i]}{\ell_{u:t}[j]} \\
&=\; \frac{p\bigl(y_{u:t},\, z_t = i \mid z_{u-1} = j\bigr)}
         {p\bigl(y_{u:t} \mid z_{u-1} = j\bigr)} \\
&=\; p\bigl(z_t = i \mid z_{u-1} = j,\, y_{u:t}\bigr).
\end{aligned}
$$

In particular, each row of $L_{u:t}$ is a probability distribution over the states, which means that its entries lie in $[0,1]$ and the rows sum to $1$ no matter how long the sequence of states is. There is no underflow issue anymore. $\log(l_{u:t})$ on the other hand can be computed preventing underflows, see below.

$F_{u:t}$ can be recovered from $(l_{u:t}, L_{u:t})$, and from $F_{u:t}$ we can compute the forward pass messages.

## Get $(\log(l), L)$ via scan

What we need to show is that we can compute $l$ and $L$ with a scan and that the operation we are scanning over is associative.

Let's define the invertible map $\phi(F_{u:t}) = (\log(l_{u:t}), L_{u:t}) = \left(\log(F_{u:t}\mathbf{1}),\,\text{diag}(F_{u:t}\mathbf{1})^{-1} F_{u:t}\right)$. The domain of the map is $\{F \in \mathbb{R}_{\ge 0}^{K \times K} : F\mathbf{1} > 0\}$.

The map is a bijection with inverse $\phi^{-1}((\log(l), L)) = \text{diag}(l) \cdot L$, and this allows to define an operator we can use for our scan:

$$
x_1 \oplus x_2 = \phi\bigl( \phi^{-1}(x_1) \phi^{-1}(x_2)\bigr).
$$

The operator is associative because the matrix product is and $\phi$ is a bijection.

We showed that $F_{u:t} = F_{u:v-1} \cdot F_{v:t}$,

$$
\begin{aligned}
(\log(l_{u:v-1}), L_{u:v-1}) \oplus (\log(l_{v:t}), L_{v:t}) &= \phi(F_{u:v-1} \cdot F_{v:t}) \\
&= \phi(F_{u:t}) = (\log(l_{u:t}), L_{u:t}).
\end{aligned}
$$

This is exactly what we need to be able to scan. A scan step computes:

$$
\begin{aligned}
\log(l_{u:t}) &= \log(F_{u:v-1} \cdot F_{v:t} \mathbf{1}) = \log(\phi^{-1}(\log(l_{u:v-1}), L_{u:v-1}) \cdot \phi^{-1}(\log(l_{v:t}), L_{v:t}) \mathbf{1}) \\
&= \log(\text{diag}(l_{u:v-1}) L_{u:v-1} \text{diag}(l_{v:t}) L_{v:t} \mathbf{1}) \\
&= \log(\text{diag}(l_{u:v-1}) L_{u:v-1} \text{diag}(l_{v:t})\mathbf{1}) \\
&\overset{\bullet}{=} \log(\text{diag}(l_{u:v-1}) L_{u:v-1} l_{v:t}) \\
&= \log(l_{u:v-1} \odot L_{u:v-1} l_{v:t}) = \log(l_{u:v-1}) + \log(L_{u:v-1} l_{v:t}).
\end{aligned}
$$

And,

$$
\begin{aligned}
L_{u:t} &= \text{diag}(l_{u:t})^{-1} F_{u:v-1} \cdot F_{v:t} \\
&\overset{\star}{=} \text{diag}(L_{u:v-1} l_{v:t})^{-1} \, \text{diag}(l_{u:v-1})^{-1} \text{diag}(l_{u:v-1}) L_{u:v-1} \text{diag}(l_{v:t}) L_{v:t} \\
&= \text{diag}(L_{u:v-1} l_{v:t})^{-1} L_{u:v-1} \text{diag}(l_{v:t}) L_{v:t}.
\end{aligned}
$$

Where $(\star)$ comes from $(\bullet)$.

If we computed $l_{v:t} = \exp\left(\log(l_{v:t})\right)$, that could underflow. The rest of the terms include matrix product involving $L_{a:b}$ and adding a $\log(l_{u:v-1})$ term, both are stable operations since $L_{a:b}$ are a row-stochastic matrices.

For numerical stability, we can set $m=\max_k \log(l_{v:t})[k]$ and $w=\exp(\log(l_{v:t}) - m\mathbf{1})$, so that $w \in (0,1]^K$ and $l_{v:t}=e^m w$ (the letter $u$ is already taken by the segment start index).

Replacing in the expression derived,

$$

\begin{cases}
\log(l_{u:t}) =  \log(l_{u:v-1}) + \log(L_{u:v-1} w) + m\mathbf{1}\\
\begin{aligned}
L_{u:t} &= \operatorname{diag}\bigl(L_{u:v-1} e^{m}w\bigr)^{-1} L_{u:v-1} \operatorname{diag}\bigl(e^{m}w\bigr) L_{v:t} \\
        &= e^{-m} e^{m}\, \operatorname{diag}\bigl(L_{u:v-1} w\bigr)^{-1} L_{u:v-1}  \operatorname{diag}(w)\, L_{v:t}\\
        &= \operatorname{diag}\bigl(L_{u:v-1} w\bigr)^{-1} L_{u:v-1}  \operatorname{diag}(w)\, L_{v:t}
\end{aligned}
\end{cases}
$$
## The Scan, Step by Step

An element is a pair $x = (\log l, L)$ with $L$ row-stochastic, composed with earlier segments on the left. Every operation below is one of

$$
\text{cond}(M, \log w) \;=\; \left(\log(Mw), \;\; \text{diag}(Mw)^{-1} M \,\text{diag}(w)\right), \qquad M \text{ row-stochastic}, \; w > 0,
$$

which conditions a row-stochastic matrix on a nonnegative weight per exit state and renormalizes the rows, returning the log row sums it divided out. Its output is again row-stochastic.

### Elements

Applying $\phi$ to Proposition 1, $x_t = \phi(F_{t:t})$ is

$$
x_t = \begin{cases}
\left(\log(\pi^\top b_t)\,\mathbf{1}, \;\; \mathbf{1}(\pi \odot b_t)^\top / (\pi^\top b_t)\right), & t \in S, \\
\left(\log(A b_t), \;\; \text{diag}(A b_t)^{-1} A\, \text{diag}(b_t)\right), & t \notin S,
\end{cases}
$$

that is, $x_t = \text{cond}(\mathbf{1}\pi^\top, \log b_t)$ at a session start and $x_t = \text{cond}(A, \log b_t)$ elsewhere. Sessions enter only through which matrix is conditioned: one `where` on the base matrix, then one batched $\text{cond}$ over all $t$.

### Combine

With $m = \max_k \log l_2[k]$ and $w = \exp(\log l_2 - m \mathbf{1}) \in (0,1]^K$, the stabilized formulas of the previous section read

$$
x_1 \oplus x_2 \;=\; \left(\log l_1 + \log(L_1 w) + m\mathbf{1}, \;\; \text{diag}(L_1 w)^{-1} L_1 \text{diag}(w)\, L_2 \right),
$$

which is $\text{cond}(L_1, \log l_2 - m\mathbf{1})$, giving $(\log r, L')$, followed by $\log l_1 + \log r + m\mathbf{1}$ and $L' L_2$. Nothing in the step can overflow: $w \le 1$, $L_1 w$ is a convex combination of entries of $w$ and so lies in $(0,1]$, and the matrix half is row-stochastic again by construction rather than by accumulated luck.

### Dropping the accumulated scale

For $c>0$, $\phi(cF) = (\log l + \log c\,\mathbf{1}, L)$: a common scalar factor lands entirely in the log half and leaves every $L$ untouched. Subtracting $\max_k \log l[k]$ from an element after each combine therefore changes nothing that is read out, and it keeps $\log l_{0:t}$ from accumulating the segment log-likelihood, which grows linearly in $t$, while the combine reads only its $O(1)$ internal differences. The price is that $\log l_{0:T-1}$ is no longer $\log p(y_{0:T-1})$, which is why the normalizers are recomputed below instead of being differenced out of it.

### Read-out

$L_{0:t}$ is $F_{0:t}$ with its rows divided by their sums, so by Proposition 3 every row of it is $\hat{\alpha}_t$:

$$
\hat{\alpha}_t = L_{0:t}[j, \cdot] \quad \text{for any } j,
$$

and the implementation takes row $0$. No reset flag is needed for this: index $0$ is a session start, so $L_{0:0}$ has all rows equal, and both right-multiplication and the row rescaling in $\oplus$ preserve equal rows, so $L_{0:t}$ has equal rows for every $t$ by induction.

### Per-step normalizers

The scan does not produce $c_t$, and `_backward_pass` consumes it. Differencing the accumulated scale, $\log c_t = \log l_{0:t} - \log l_{0:t-1}$, is exactly the cancellation the parametrization was built to avoid. Recompute it locally instead: once all $\hat{\alpha}_t$ are known, the forward recursion gives

$$
c_t = \begin{cases}
\left((\hat{\alpha}_{t-1} A) \odot b_t\right)\mathbf{1}, & t \notin S, \\
\left(\pi \odot b_t\right)\mathbf{1}, & t \in S,
\end{cases}
$$

for all $t$ at once — one $T \times K$ by $K \times K$ product and one reduction, every operand $O(1)$. The dummy $\hat{\alpha}_{-1}$ that makes the shapes line up at $t=0$ is discarded by the $t \in S$ branch, but it must be a valid distribution rather than zeros: $\log 0$ has an infinite derivative, and reverse-mode differentiation of a `where` hands that infinity a zero cotangent, producing `nan`.

### Sketch

```python
def cond(M, log_w):
    """Condition a row-stochastic M on log weights per exit state, renormalizing rows."""
    m = jnp.max(log_w, axis=-1, keepdims=True)
    Mw = M * jnp.exp(log_w - m)[..., None, :]      # diag-free M @ diag(w)
    r = jnp.sum(Mw, axis=-1)
    return jnp.log(r) + m, Mw / r[..., None]

def combine(x1, x2):                                # x1 earlier, x2 later
    log_l1, L1 = x1
    log_l2, L2 = x2
    log_r, L = cond(L1, log_l2)
    log_l = log_l1 + log_r
    return log_l - jnp.max(log_l, axis=-1, keepdims=True), L @ L2

def forward(log_pi, log_A, log_b, session_starts):
    base = jnp.where(session_starts[:, None, None], jnp.exp(log_pi), jnp.exp(log_A))
    elements = cond(base, log_b)                    # x_t = phi(F_{t:t}), batched over t
    _, L_cum = jax.lax.associative_scan(combine, elements)
    alphas = L_cum[:, 0, :]                         # rows of L_{0:t} are all equal
    return jnp.log(alphas), normalizers(log_pi, log_A, log_b, session_starts, alphas)
```

`jnp.exp(log_pi)` broadcasts against a $(T, K, K)$ base, which is the $\mathbf{1}\pi^\top$ of the element formula.

### Cost

Each $\oplus$ is two $K \times K$ matrix products plus $O(K^2)$ elementwise work, and `associative_scan` performs about $2T$ of them: $O(TK^3)$ work at depth $O(\log T)$, against $O(TK^2)$ work at depth $O(T)$ for the sequential recursion. The trade is worth taking only where the $T$ sequential steps, each of them a small kernel launch, dominate — which is what the benchmarks have to decide, per $K$ and per device.

The backward pass is not covered by the propositions above. Its messages are not row-stochastic (they carry the $1/c_t$ factors), the equal-rows argument that made sessions free here does not apply to it, and its elements need an explicit reset flag.
