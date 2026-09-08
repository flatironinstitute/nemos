"""PP-GLM input data definitions and type aliases."""

import equinox as eqx
from jaxtyping import Array, Float, Int


class PredictorsPPGLM(eqx.Module):
    """Preprocessed predictors for PP-GLM."""

    times: Float[Array, "n_events"]  # noqa: F821
    predictor_ids: Int[Array, "n_events"]  # noqa: F821


class SpikesPPGLM(eqx.Module):
    """Preprocessed spikes for PP-GLM."""

    times: Float[Array, "n_spikes"]  # noqa: F821
    neuron_ids: Int[Array, "n_spikes"]  # noqa: F821
    timestamp_idx: Int[Array, "n_spikes"]  # noqa: F821


class MCSamplePPGLM(eqx.Module):
    """Preprocessed Monte Carlo sample points for PP-GLM."""

    times: Float[Array, "n_samples"]  # noqa: F821
    timestamp_idx: Int[Array, "n_samples"]  # noqa: F821
