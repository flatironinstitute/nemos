"""PP-GLM input data definitions and type aliases."""

import equinox as eqx
from jaxtyping import Array, Float, Int

class X_ppglm(eqx.Module):
    """Preprocessed predictors for PP-GLM."""

    times: Float[Array, "n_events"]
    ids: Int[Array, "n_events"]


class y_ppglm(eqx.Module):
    """Preprocessed spikes for PP-GLM."""

    times: Float[Array, "n_spikes"]
    ids: Int[Array, "n_spikes"]
    idx: Int[Array, "n_spikes"]


class mc_sample_ppglm(eqx.Module):
    """Preprocessed Monte Carlo sample points for PP-GLM."""

    times: Float[Array, "n_samples"]
    idx: Int[Array, "n_samples"]