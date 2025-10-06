"""
Simulate population GLM with hidden Markov states.

This script generates synthetic neural data where each neuron's activity
depends on both external features and a latent state that evolves over time
according to a Markov process.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import click
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import nemos as nmo


def parse_int_list(ctx, param, value):
    if value is None:
        return [10]
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise click.BadParameter("Must be comma-separated integers")


def load_base_parameters(data_path: str) -> Dict[str, NDArray]:
    """
    Load baseline parameters from data file.

    Parameters
    ----------
    data_path
        Path to the data file

    Returns
    -------
    Dictionary containing initial_prob, transition_matrix, orig_weights, design_matrix, and new_sess
    """
    data = np.load(data_path)
    return {
        "initial_prob": data["initial_prob"],
        "transition_prob": data["transition_prob"],
        "orig_weights": data["projection_weights"][1:],
        "design_matrix": data["X"],
        "new_sess": data["new_sess"],
    }


def create_projection_weights(
    orig_weights: NDArray, n_neurons: int, n_states: int, seed: int = 123
) -> jnp.ndarray:
    """
    Create neuron-specific latent weights for each state.

    Parameters
    ----------
    orig_weights
        Base weights of shape (n_features, n_states)
    n_neurons
        Number of neurons to simulate
    n_states
        Number of latent states
    seed
        Random seed for reproducibility

    Returns
    -------
    projection_weights
        Weights of shape (n_features+1, n_neurons, n_states)
        First feature is the intercept/state-selection term
    """
    n_features = orig_weights.shape[0]
    projection_weights = jnp.zeros((n_features + 1, n_neurons, n_states))

    key = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    for j in range(n_neurons):
        # Add neuron-specific noise to base weights
        key, subkey = jax.random.split(key)
        noisy_weights = orig_weights + jax.random.normal(
            subkey, shape=orig_weights.shape
        )
        projection_weights = projection_weights.at[1:, j, :].set(noisy_weights)

        # Each neuron is "active" in one state and suppressed in others
        active_state = np.random.choice(n_states)
        inactive_states = np.arange(n_states)[np.arange(n_states) != active_state]

        projection_weights = projection_weights.at[0, j, inactive_states].set(-20)
        projection_weights = projection_weights.at[0, j, active_state].set(2)

    return projection_weights


def simulate_hmm_glm(
    design_matrix: NDArray,
    projection_weights: jnp.ndarray,
    transition_prob: NDArray,
    initial_prob: NDArray,
    seed: int = 123,
) -> Dict[str, NDArray]:
    """
    Simulate population GLM with hidden Markov states.

    Parameters
    ----------
    design_matrix
        Design matrix of shape (n_timepoints, n_features)
    projection_weights
        Weights of shape (n_features+1, n_neurons, n_states)
    transition_prob
        State transition probabilities of shape (n_states, n_states)
    initial_prob
        Initial state probabilities of shape (n_states,)
    seed
        Random seed

    Returns
    -------
    dict with keys:
        'counts' : array of shape (n_timepoints, n_neurons)
        'rates' : array of shape (n_timepoints, n_neurons)
        'latent_states' : array of shape (n_timepoints, n_states)
    """
    n_timepoints = design_matrix.shape[0]
    n_neurons = projection_weights.shape[1]
    n_states = projection_weights.shape[2]

    # Initialize GLM
    glm = nmo.glm.PopulationGLM(observation_model="Bernoulli")
    glm.intercept_ = jnp.zeros((n_neurons,))

    # Initialize storage
    latent_states = np.zeros((n_timepoints, n_states), dtype=int)
    rates = np.zeros((n_timepoints, n_neurons))
    counts = np.zeros((n_timepoints, n_neurons))

    # Sample initial state
    np.random.seed(seed)
    initial_state = np.random.choice(n_states, p=initial_prob)
    latent_states[0, initial_state] = 1

    # Set initial weights and simulate first timepoint
    glm.coef_ = projection_weights[..., initial_state].reshape(
        projection_weights.shape[0], n_neurons
    )
    glm._initialize_feature_mask(design_matrix, rates)

    key = jax.random.PRNGKey(seed)
    counts[0], rates[0] = glm.simulate(key, design_matrix[:1])

    # Simulate remaining timepoints
    for t in range(1, n_timepoints):
        # Sample next state
        key, subkey = jax.random.split(key)
        prev_state_vec = latent_states[t - 1]
        transition_probs = transition_prob.T @ prev_state_vec
        next_state = jax.random.choice(subkey, jnp.arange(n_states), p=transition_probs)
        latent_states[t, next_state] = 1

        # Update weights and simulate
        glm.coef_ = projection_weights[..., next_state]
        key, subkey = jax.random.split(key)
        counts[t], rates[t] = glm.simulate(subkey, design_matrix[t : t + 1])

    return {"counts": counts, "rates": rates, "latent_states": latent_states}


def run_simulation(
    n_neurons: int, data_path: Optional[str] = None, seed: int = 123
) -> Dict:
    """
    Run complete simulation pipeline.

    Parameters
    ----------
    n_neurons
        Number of neurons to simulate
    data_path
        Path to data file. If None, fetches default data.
    seed
        Random seed for reproducibility

    Returns
    -------
    dict containing simulation results and parameters
    """
    # Load base parameters
    if data_path is None:
        data_path = nmo.fetch.fetch_data("em_three_states.npz")

    params = load_base_parameters(data_path)
    n_states = params["transition_prob"].shape[0]

    # Create neuron-specific weights
    projection_weights = create_projection_weights(
        params["orig_weights"], n_neurons, n_states, seed=seed
    )

    # Run simulation
    results = simulate_hmm_glm(
        params["design_matrix"],
        projection_weights,
        params["transition_prob"],
        params["initial_prob"],
        seed=seed,
    )

    # Package results
    return {
        **results,
        "projection_weights": projection_weights,
        "transition_prob": params["transition_prob"],
        "initial_prob": params["initial_prob"],
        "design_matrix": params["design_matrix"],
        "n_neurons": n_neurons,
        "n_states": n_states,
    }


@click.command()
@click.option(
    "--n-neurons",
    callback=parse_int_list,
    default="10",
    help="Number of neurons to simulate (comma-separated: 2,3,4)",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to data file. If not provided, fetches default data.",
)
@click.option("--seed", type=int, default=123, help="Random seed for reproducibility")
@click.option(
    "--output-directory",
    type=click.Path(),
    default=None,
    help="Path to save output .npz file. If not provided, use current directory.",
)
@click.option(
    "--base-name",
    type=click.Path(),
    default="glm_hmm_simulation",
    help="Base name for the output .npz file. The Full name will be `base_name_n_neurons_{n_neurons}_seed_{seed}.npz`.}`",
)
@click.option(
    "--verbose/--no-verbose",
    "-v/-V",  # Add short flags
    default=True,
    help="Print simulation details",
)
def main(
    n_neurons: List[int],
    data_path: Optional[str],
    seed: int,
    output_directory: Optional[str],
    base_name: Optional[str],
    verbose: bool,
):
    """
    Run GLM simulation with hidden Markov states.

    Example usage:

        python script.py --n-neurons 20 --seed 42

        python script.py --n-neurons 50 --output results.npz
    """
    if output_directory is None:
        output_directory = Path(
            os.environ.get("NEMOS_DATA_DIR", Path(__file__))
        ).resolve()
    else:
        output_directory = Path(output_directory).resolve()

    print(f"Saving to directory: {output_directory}")
    output_directory.mkdir(exist_ok=True, parents=True)

    for neurons in n_neurons:
        output = output_directory / (
            Path(base_name).stem + f"_n_neurons_{neurons}_seed_{seed}.npz"
        )
        if output.exists():
            print("Output file already exists, skipping.")
            continue

        results = run_simulation(n_neurons=neurons, data_path=data_path, seed=seed)

        if verbose:
            click.echo(
                f"Simulated {n_neurons} neurons over {results['counts'].shape[0]} timepoints"
            )
            click.echo(f"Number of states: {results['n_states']}")
            click.echo(f"Counts shape: {results['counts'].shape}")
            click.echo(f"Rates shape: {results['rates'].shape}")
            click.echo(f"Latent states shape: {results['latent_states'].shape}")

        np.savez(
            output,
            counts=results["counts"],
            rates=results["rates"],
            latent_states=results["latent_states"],
            projection_weights=jnp.squeeze(results["projection_weights"]),
            transition_prob=results["transition_prob"],
            initial_prob=results["initial_prob"],
            design_matrix=results["design_matrix"],
        )
        if verbose:
            click.echo(f"Results saved to {output}")


if __name__ == "__main__":
    main()
