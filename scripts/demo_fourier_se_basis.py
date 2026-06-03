"""Demo: FourierSEBasis — Fourier basis with squared-exponential implied prior.

Constructs a 1-D Fourier basis on a chosen domain whose columns are weighted
so that an i.i.d. ``N(0, 1)`` prior on the basis coefficients corresponds to a
Gaussian process with squared-exponential covariance. Reproduces the spirit
of ``efgp_jax/examples/gp_discretization_1d.py`` using ``FourierSEBasis``.

The script:
    1. Builds a ``FourierSEBasis``.
    2. Verifies that ``Phi @ Phi.T`` recovers the SE kernel.
    3. Draws prior samples via ``f(x) = Phi(x) @ z`` with ``z ~ N(0, I)``.
    4. Plots the first few basis functions, the implied vs exact kernel, and
       prior samples.

Usage:
    python scripts/demo_fourier_se_basis.py
"""

import os

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nemos.basis import FourierSEBasis

# ---------------------------------------------------------------------------
# 1. Build the basis
# ---------------------------------------------------------------------------
domain = (0.0, 1.0)
lengthscale = 0.2
variance = 1.0
eps = 1e-2

basis = FourierSEBasis(
    lengthscale=lengthscale, domain=domain, eps=eps, variance=variance
)
print(f"FourierSEBasis: lengthscale={lengthscale}, domain={domain}, eps={eps}")
print(f"  m = {basis._m}, h = {basis._h:.4f}, n_basis_funcs = {basis.n_basis_funcs}")
print(f"  xis = {np.asarray(basis.xis)}")


# ---------------------------------------------------------------------------
# 2. Implied covariance vs SE kernel
# ---------------------------------------------------------------------------
x = jnp.linspace(domain[0], domain[1], 400)
Phi = np.asarray(basis.evaluate(x))  # (N, n_basis_funcs)
K_impl = Phi @ Phi.T  # implied covariance under N(0, I) prior

x_np = np.asarray(x)
i_mid = len(x_np) // 2
r = x_np - x_np[i_mid]
K_exact = variance * np.exp(-(r**2) / (2 * lengthscale**2))
K_slice = K_impl[i_mid]

print(
    f"\nmax |K_impl - K_SE| (interior, |r| < L/2): "
    f"{np.max(np.abs(K_slice[np.abs(r) < 0.5] - K_exact[np.abs(r) < 0.5])):.3e}"
)


# ---------------------------------------------------------------------------
# 3. Prior samples: f(x) = Phi(x) @ z,  z ~ N(0, I)
# ---------------------------------------------------------------------------
n_samples = 5
key = jax.random.PRNGKey(0)
samples = np.asarray(basis.sample(x, key, n_samples=n_samples))  # (n_samples, N)


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
n_show = 4
for k in range(n_show):
    ax.plot(x_np, Phi[:, k], label=f"col {k}")
ax.set_title(f"First {n_show} basis columns (cos × weights)")
ax.set_xlabel("x")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(r, K_exact, "k-", lw=2, label="exact SE")
ax.plot(r, K_slice, "r--", lw=1.5, label=f"Phi Phi^T (M={basis.n_basis_funcs})")
ax.set_title("Implied covariance vs SE kernel")
ax.set_xlabel("r = x - x_mid")
ax.set_ylabel("k(r)")
ax.legend(fontsize=9)

ax = axes[2]
for s in range(n_samples):
    ax.plot(x_np, samples[s], lw=0.9, alpha=0.85)
ax.set_title(f"Prior samples (l={lengthscale}, var={variance})")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")

fig.suptitle(
    f"FourierSEBasis demo:  domain={domain}, l={lengthscale}, var={variance}, "
    f"eps={eps:g}, M={basis.n_basis_funcs}",
    y=1.02,
)
fig.tight_layout()

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "demo_fourier_se_basis.png"
)
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
