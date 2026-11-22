"""Samplers for drawing input values to simulate GPGLM-family models."""

import jax
import jax.numpy as jnp
from jax import lax


class Roberts:
    """Quasi-random sampler tiling the hypercube ``[0, 1]^d``, where ``d`` is ``ndim``.

    Generates a low-discrepancy (Roberts) sequence: points spread evenly over the
    hypercube, covering it more uniformly than independent uniform samples. The
    sequence is shifted by a random offset, so different random keys give different,
    equally well-spread point sets.

    Code modified from:
    https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab

    Parameters
    ----------
    ndim :
        Number of dimensions ``d`` of the hypercube. Default is 1.
    mc_scale :
        Scale applied to the samples, which then tile ``[0, mc_scale]^d``.
        Default is 1.
    root_iters :
        Number of Newton-Raphson iterations used to compute the root that defines
        the sequence. Default is 10,000.

    Attributes
    ----------
    basis :
        Per-dimension step of the sequence, shape ``(ndim,)``.
    """

    def __init__(self, ndim=1, mc_scale=1, root_iters=10_000):

        self.ndim = ndim
        self.root_iters = root_iters
        self.mc_scale = mc_scale

        root = self._compute_root()
        # Compute basis parameter
        self.basis = 1 - (1 / root ** (1 + jnp.arange(self.ndim)))

    # Compute the unique positive root of f using the Newton-Raphson method.
    def f(self, x):
        """Polynomial whose unique positive root defines the sequence, ``x^(d+1) - x - 1``.

        Parameters
        ----------
        x :
            Input value.

        Returns
        -------
        :
            The value of the polynomial at ``x``.
        """
        return x ** (self.ndim + 1) - x - 1

    def _compute_root(self):
        """Compute the unique positive root of ``f`` using the Newton-Raphson method.

        Returns
        -------
        :
            The positive root of ``f``, starting the iterations from 1.0.
        """

        def newton_raphson_update(x, _):
            f_val = self.f(x)
            f_grad = jax.grad(self.f)(x)
            return x - f_val / f_grad, None

        # Perform Newton-Raphson iterations using lax.scan
        # TODO - Something is deprecated here! Update jax. argument lengths should take the place of length of xs if no
        # xs is needed, but getting an error if not included
        root, _ = lax.scan(
            newton_raphson_update,
            1.0,
            xs=jnp.arange(self.root_iters),
            lengths=self.root_iters,
        )
        return root

    def sample(self, random_key, n_points):
        """Draw points from the randomly shifted Roberts sequence.

        Parameters
        ----------
        random_key :
            jax.random.key used to draw the random offset of the sequence.
        n_points :
            Number of points to draw.

        Returns
        -------
        :
            The sampled points, shape ``(n_points, ndim)``, with values in
            ``[0, mc_scale)``.
        """
        # Define sequence without taking modulo 1
        sequence = jnp.arange(n_points)[:, None] * self.basis[None, :]

        # TODO - issue with random seeding - self consistent but doesnt match demo colab,
        # even with same seed. sequence etc match
        # print(jax.random.uniform(jax.random.PRNGKey(123), shape=[self.ndim]))
        return (
            jnp.modf(sequence + jax.random.uniform(random_key, shape=[self.ndim]))[0]
            * self.mc_scale
        )


class RandomTrajectory1D:
    """Random walk on a 1D interval with reflecting boundaries.

    Simulates a trajectory, such as an animal moving along a linear track, that moves
    at a constant velocity, randomly reverses direction, and bounces off the ends of the
    interval ``[0, mc_scale]``.

    Parameters
    ----------
    velocity :
        Distance moved per step. Default is 0.01.
    mc_scale :
        Upper end of the interval. Default is 1.

    Attributes
    ----------
    high :
        Upper boundary of the interval.
    low :
        Lower boundary of the interval, always 0.
    """

    def __init__(self, velocity=0.01, mc_scale=1):
        self.velocity = velocity
        self.high = 1 * mc_scale
        self.low = 0

    def sample(self, key, lengths, x0=0.5, flip_prob=0.05):
        """Simulate a trajectory.

        Parameters
        ----------
        key :
            jax.random.key for seeding the simulation.
        lengths :
            Number of time steps in the trajectory, including the starting position.
        x0 :
            Starting position. Default is 0.5.
        flip_prob :
            Probability of reversing direction at each step. Default is 0.05.

        Returns
        -------
        :
            The positions over time, shape ``(lengths,)``.

        """
        key_dir, key_scan = jax.random.split(key)
        direction0 = jax.random.choice(key_dir, jnp.array([-1.0, 1.0]))

        def step(carry, subkey):
            pos, direction = carry
            flip = jax.random.bernoulli(subkey, flip_prob)
            direction = jnp.where(flip, -direction, direction)

            new_pos = pos + direction * self.velocity
            over_high, under_low = new_pos > self.high, new_pos < self.low
            new_pos = jnp.where(over_high, 2 * self.high - new_pos, new_pos)
            new_pos = jnp.where(under_low, 2 * self.low - new_pos, new_pos)
            direction = jnp.where(over_high | under_low, -direction, direction)
            return (new_pos, direction), new_pos

        keys = jax.random.split(key_scan, lengths - 1)
        (_, _), positions = jax.lax.scan(step, (x0, direction0), keys)
        return jnp.concatenate([jnp.array([x0]), positions])
