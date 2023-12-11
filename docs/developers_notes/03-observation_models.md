# The `observation_models` Module

## Introduction

The `observation_models` module provides objects representing the observations of GLM-like models.

The abstract class `Observations` defines the structure of the subclasses which specify observation types, such as Poisson, Gamma, etc. These objects serve as attributes of the [`nemos.glm.GLM`](../05-glm/#the-concrete-class-glm) class, equipping the GLM with a negative log-likelihood. This is used to define the optimization objective, the deviance which measures model fit quality, and the emission of new observations, for simulating new data.

## The Abstract class `Observations`

The abstract class `Observations` is the backbone of any observation model. Any class inheriting `Observations` must reimplement the `negative_log_likelihood`, `sample_generator`, `residual_deviance`, and `estimate_scale` methods.

### Abstract Methods

For subclasses derived from `Observations` to function correctly, they must implement the following:

- **negative_log_likelihood**: Computes the negative-log likelihood of the model up to a normalization constant. This method is usually part of the objective function used to learn GLM parameters.
  
- **sample_generator**: Returns the random emission probability function. This typically invokes `jax.random` emission probability, provided some sufficient statistics[^1]. For distributions in the exponential family, the sufficient statistics are the canonical parameter and the scale. In GLMs, the canonical parameter is entirely specified by the model's weights, while the scale is either fixed (i.e., Poisson) or needs to be estimated (i.e., Gamma).
  
- **residual_deviance**: Computes the residual deviance based on the model's estimated rates and observations.

- **estimate_scale**: A method for estimating the scale parameter of the model.

### Public Methods

- **pseudo_r2**: Method for computing the pseudo-$R^2$ of the model based on the residual deviance. There is no consensus definition for the pseudo-$R^2$, what we used here is the definition by Cohen at al. 2003[^2]. 


### Auxiliary Methods

- **_check_inverse_link_function**: Check that the provided link function is a `Callable` of the `jax` namespace.

## Concrete `PoissonObservations` class

The `PoissonObservations` class extends the abstract `Observations` class to provide functionalities specific to the Poisson observation model. It is designed for modeling observed spike counts based on a Poisson distribution with a given rate.

### Overridden Methods

- **negative_log_likelihood**: This method computes the Poisson negative log-likelihood of the predicted rates for the observed spike counts.
  
- **sample_generator**: Generates random numbers from a Poisson distribution based on the given `predicted_rate`.
  
- **residual_deviance**: Calculates the residual deviance for a Poisson model.
  
- **estimate_scale**: Assigns a fixed value of 1 to the scale parameter of the Poisson model since Poisson distribution has a fixed scale.

## Contributor Guidelines 

To implement an observation model class you

- **Must** inherit from `Observations`

- **Must** provide a concrete implementation of `negative_log_likelihood`, `sample_generator`, `residual_deviance`, and `estimate_scale`.

- **Should not** reimplement the `pseudo_r2` method as well as the `_check_inverse_link_function` auxiliary method.

[^1]: 
    In statistics, a statistic is sufficient with respect to a statistical model and its associated unknown parameters if "no other statistic that can be calculated from the same sample provides any additional information as to the value of the parameters", adapted from Fisher R. A.
    1922. On the mathematical foundations of theoretical statistics. *Philosophical Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character* 222:309–368. http://doi.org/10.1098/rsta.1922.0009.
[^2]:
    Jacob Cohen, Patricia Cohen, Steven G. West, Leona S. Aiken. 
    *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*. 
    3rd edition. Routledge, 2002. p.502. ISBN 978-0-8058-2223-6. (May 2012)
