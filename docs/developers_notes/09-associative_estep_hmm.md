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
\tilde{\alpha}_t = (\hat{\alpha}_{t-1} A) \odot b_t \qquad c_t  = \tilde{\alpha}_t \mathbf{1} \qquad \hat{\alpha}_t / c_t.
$$
