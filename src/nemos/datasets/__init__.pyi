"""Lazy import of data generating functions."""

from ._simulated import tuned_bernoulli, tuned_gamma, tuned_gaussian, tuned_poisson

__all__ = [
    "tuned_bernoulli",
    "tuned_poisson",
    "tuned_gamma",
    "tuned_gaussian",
]
