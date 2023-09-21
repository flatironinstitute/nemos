# The `noise_model` Module

## Introduction

The `noise_model` module provides objects representing the observation noise of GLM-like models. 

The abstract class `NoiseModel` defines the structure of the subclasses which specify observation noise type, e.g. Poisson, Gamma, etc.

These objects are attributes of the `neurostatslib.glm.GLM` class, equipping the GLM with a negative log-likelihood, used to define the optimization objective, the deviance which measures model fit quality, and the emission of new observations, for simulating new data.


## The Abstract class `NoiseModel`

The abstract class `NoiseModel` is the backbone of any noise model. Any class inherting `NoiseModel` must reimplement the `negative_log_likelihood`, `emission_probability`, `residual_deviance`, and `get_scale` methods.

### Abstract Methods

For subclasses derived from `NoiseModel` to function correctly, they must implement the following:

- **negative_log_likelihood**: The negative-loglikelihood of the model. This is usually part of the objective function used to learn GLM parameters.
- **emission_probability**: The random emission probability function. Usually calls `jax.random` emission probability with some provided some sufficient statistics. For distributions in the exponential family, sufficient statistics are the canonical parameter and the scale. In GLMs, the canonical parameter is fully specified by the weights of the model, while the scale is either fixed (i.e. Poisson) or needs to be estimated (i.e. Gamma).
- **residual_deviance**: Compute the residual deviance given the current model estimated rates and observations.
- **estimate_scale**: Method for estimating the scale parameter of the model. This is required when generating simulated activity in the proper scale.

### Public Methods

- **pseudo_r2**: Method for computing the pseudo-$R^2$ of the model based on the residual deviance. There is no consensus definition for the pseudo-$R^2$, what we used here is the definition by Choen at al. 2003[^1]. 


### Auxiliary Methods

- **_check_inverse_link_function**: Check that the provided link function is a `Callable` of the `jax` namespace.

## Contributor Guidelines


[^1]:
    Jacob Cohen, Patricia Cohen, Steven G. West, Leona S. Aiken. 
    *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*. 
    3rd edition. Routledge, 2002. p.502. ISBN 978-0-8058-2223-6. (May 2012)
