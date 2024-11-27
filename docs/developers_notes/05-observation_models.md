# The `observation_models` Module

## Introduction

The [`observation_models`](observation_models) module provides objects representing the observations of GLM-like models.

The abstract class [`Observations`](nemos.observation_models.Observations) defines the structure of the subclasses which specify observation types, such as Poisson, Gamma, etc. These objects serve as attributes of the [`nemos.glm.GLM`](the-concrete-class-glm) class, equipping the GLM with a negative log-likelihood. This is used to define the optimization objective, the deviance which measures model fit quality, and the emission of new observations, for simulating new data.

(the-abstract-class-observations)=
## The Abstract class `Observations`

The abstract class [`Observations`](nemos.observation_models.Observations) is the backbone of any observation model. Any class inheriting [`Observations`](nemos.observation_models.Observations) must reimplement the `_negative_log_likelihood`, [`log_likelihood`](nemos.observation_models.Observations.log_likelihood), [`sample_generator`](nemos.observation_models.Observations.sample_generator), [`deviance`](nemos.observation_models.Observations.deviance), and [`estimate_scale`](nemos.observation_models.Observations.estimate_scale) methods.

### Abstract Methods

For subclasses derived from [`Observations`](nemos.observation_models.Observations) to function correctly, they must implement the following:

- **_negative_log_likelihood**: Computes the negative-log likelihood of the model up to a normalization constant. This method is usually part of the objective function used to learn GLM parameters.

- [`log_likelihood`](nemos.observation_models.Observations.log_likelihood): Computes the full log-likelihood including the normalization constant.
  
- [`sample_generator`](nemos.observation_models.Observations.sample_generator): Returns the random emission probability function. This typically invokes `jax.random` emission probability, provided some sufficient statistics, [see below](#suff-stat). For distributions in the exponential family, the sufficient statistics are the canonical parameter and the scale. In GLMs, the canonical parameter is entirely specified by the model's weights, while the scale is either fixed (i.e., Poisson) or needs to be estimated (i.e., Gamma).
  
- [`deviance`](nemos.observation_models.Observations.deviance): Computes the deviance based on the model's estimated rates and observations.

- [`estimate_scale`](nemos.observation_models.Observations.estimate_scale): A method for estimating the scale parameter of the model. Rate and scale are sufficient to fully characterize distributions from the exponential family.

:::{dropdown} <span id="suff-stat">Sufficient statistics</span>
:color: info
:icon: info

In statistics, a statistic is sufficient with respect to a statistical model and its associated unknown parameters if "no other statistic that can be calculated from the same sample provides any additional information as to the value of the parameters", adapted from [[1]](#ref-1).
:::

### Public Methods

- [`pseudo_r2`](nemos.observation_models.Observations.pseudo_r2): Method for computing the pseudo-$R^2$ of the model based on the residual deviance. There is no consensus definition for the pseudo-$R^2$, what we used here is the definition by Cohen at al. 2003[$^{[2]}$](#ref-2). 
- [`check_inverse_link_function`](nemos.observation_models.Observations.check_inverse_link_function): Check that the link function is a auto-differentiable, vectorized function form $\mathbb{R} \longrightarrow \mathbb{R}$.

## Contributor Guidelines 

To implement an observation model class you

- **Must** inherit from [`Observations`](nemos.observation_models.Observations)

- **Must** provide a concrete implementation of the abstract methods, see above.

- **Should not** reimplement the [`pseudo_r2`](nemos.observation_models.Observations.pseudo_r2) method as well as the [`check_inverse_link_function`](nemos.observation_models.Observations.check_inverse_link_function) auxiliary method.

## References
<p id="ref-1">[1]
    <a href="https://doi.org/10.1098/rsta.1922.0009">Fisher R. A. 1922. On the mathematical foundations of theoretical statistics. <em>Philosophical Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character</em> 222:309–368. http://doi.org/10.1098/rsta.1922.0009</a>.
</p>

<p id="ref-2">
    [2] 
    <a href="https://www.taylorfrancis.com/books/mono/10.4324/9780203774441/applied-multiple-regression-correlation-analysis-behavioral-sciences-jacob-cohen-patricia-cohen-stephen-west-leona-aiken">
        Jacob Cohen, Patricia Cohen, Steven G. West, Leona S. Aiken. <em>Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences</em>. 3rd edition. Routledge, 2002. p.502. ISBN 978-0-8058-2223-6. (May 2012)
    </a>
</p>
