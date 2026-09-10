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

## Scan Map

Let $\mathcal{M}$ be a set. For $x=(x_0,...,x_{T-1})\in \mathcal{M}^T$ and let $X_{a:b} = x_a \oplus \dots \oplus x_b$. We can define a *scan map* as

$$
\mathcal{S}: \mathcal{M}^T \rightarrow \mathcal{M}^T, \;\;\; \mathcal{S}(x) = (X_{0:0}, \dots X_{0:T-1})
$$

If $\oplus$ be an associative operator on $\mathcal{M}$, then the scan map can be computed efficiently via `jax.lax.associative_scan`.

## The forward pass associativity

The forward step computes two quantities:

$$
\hat{\alpha}_t[i] \;=\; p\bigl(z_t = i \mid y_{s(t):t}\bigr), \qquad c_t \;=\; p\bigl(y_t \mid y_{s(t):t-1}\bigr),
$$

These quantities can be computed recursively as follows:

$$
\begin{equation}
\tilde{\alpha}_t = (\hat{\alpha}_{t-1} A) \odot b_t \qquad c_t  = \tilde{\alpha}_t \mathbf{1} \qquad \hat{\alpha}_t / c_t.
\end{equation}
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
    \mathbf{1} (\pi \odot b_t) & t \in S, \\
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

By definition of $F_{u:t}$ and the Bayes theorem we have:

$$
\begin{aligned}
F_{s(t):t}[j,\cdot] &= p(y_{s(t):t}, z_t = \cdot) \\
&= p(y_{s(t):t}) \frac{p(y_{s(t):t}, z_t = \cdot)}{p(y_{s(t):t})}\\
&= p(y_{s(t):t}) p(z_t = \cdot | y_{s(t):t})\\
&= p(y_{s(t):t}) \hat{\alpha}_t[\cdot],
\end{aligned}
$$

Since $s(t) \in S$.

:::
