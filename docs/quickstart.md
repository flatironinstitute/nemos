
This tutorial will introduce the main `nemos` functionalities. This is intended for users that are 
already familiar with the GLM framework but want to learn how to interact with the `nemos` API. 
If you have used [scikit-learn](https://scikit-learn.org/stable/) before, we are compatible the [estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html), so the quickstart should look 
familiar.

In the following sessions, you will learn:

1. How to define and fit a GLM model.
2. Which are the GLM parameters and hyperparameters and how to set it.
3. How `nemos` interacts with `pynapple` for pre-processing
4. How `nemos` interacts with `scikit-learn` for pipelines and cross-validation. 

Each of this section will be an independent example.

### Basic Model Fit

Defining and fitting a `nemos` GLM model is straightforward and can be done using the following syntax:

```python
import nemos as nmo
import numpy as np

# predictors, shape (n_samples, n_neurons, n_features)
X = 0.2 * np.random.normal(size=(100, 1, 1))
# true coefficients, shape (n_neurons, n_features)
coef = np.random.normal(size=(1, 1))
# observed counts, shape (n_samples, n_neurons)
y = np.random.poisson(np.exp(np.einsum("nf, tnf -> tn", coef, X)))

# model definition
model = nmo.glm.GLM()
# model fitting
model.fit(X, y)
```


Once fit, you can retrieve model parameters as follows,

```python
# model coefficients, shape (n_neurons, n_features)
print(f"Model coefficients: {model.coef_.shape}")
# out: Model coefficients: (1, 1)

# model coefficients, shape (n_neurons, )
print(f"Model intercept: {model.intercept_.shape}")
# out: Model intercept: (1,)
```

!!! note
    This works the same as in scikit-learn, see for example their [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

### Model Arguments

The `model` object has the following arguments, stored as attributes,

1. `model.observation_model`: The statistical model for the observed variable. The only available option so far is `nemos.observation_models.PoissonObservation`, which is the most common choice for modeling spike counts.
2. `model.regularizer`: Determines the regularization type, defaulting to `nemos.regularizer.Ridge`, for $L_2$ regularization.

You can change the defaults when you instantiate the model or after.

```python
import nemos as nmo

# set no regularization at initialization
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized())
print(model.regularizer)


# change to Lasso
model.regularizer = nmo.regularizer.Lasso()
print(model.regularizer)
```

#### Observation Model Hyperparameters

The observation model has a single hyper-parameter, the non-linearity which maps a linear combination of predictors 
to the neural activity mean (the instantaneous firing rate). We call the non-linearity *inverse link-function*, 
naming convention from the [statistical literature on  GLMs](https://en.wikipedia.org/wiki/Generalized_linear_model).
The default for the `PoissonObservation` is the exponential $f(x) = e^x$, implemented in JAX as `jax.numpy.exp`. 
Another common choice is the "soft-plus", in JAX this is implemented as `jax.nn.softplus`. 

As with all `nemos` objects, one can set the parameters at initialization or after.

```python
import jax
import nemos as nmo

# change default 
obs_model = nmo.observation_models.PoissonObservations(inverse_link_function=jax.nn.softplus)
model = nmo.glm.GLM(observation_model=obs_model)
print(model.observation_model.inverse_link_function)

# change back to the exponential
model.observation_model.inverse_link_function = jax.numpy.exp
print(model.observation_model.inverse_link_function)
```

These two options result in a convex optimization objective for all the provided regularizers 
(un-regularized, Ridge, Lasso, group-Lasso). This is nice because we can guarantee that there exists a single optimal 
set of model coefficients.

The user can set any non-linearity, provided that:

1. JAX can auto-differentiate it.
2. It maps scalars into scalars, i.e. $f : \mathbb{R} \longrightarrow \mathbb{R}$.
3. It is vectorized, i.e. if you pass a NumPy array or a JAX array, it returns an array of the same shape.

However, in the general case, we cannot guarantee the convexity of the objective anymore.

#### Regularizer Hyperparameters

Object of type regularizer have different hyper-parameters, depending on the type, but they all share the following:

1. `solver_name`: The name of the [`jaxopt`](https://jaxopt.github.io/stable/) solver used for learning the model parameters (for instance, 
    `GradientDescent`, `BFGS`, etc.)
2. `solver_kwargs`: Additional arguments that should be passed to the solver (for instance, `tol`, `max_iter` and 
    similar) as a dictionary.

For each regularizer, we have one or more allowable solvers, stored in the `allowed_solvers` attribute. 
This ensures that for each regularization scheme, we run an appropriate optimization algorithm. 

```python
import nemos as nmo

regularizer = nmo.regularizer.Ridge()
print(f"Allowed solver: {regularizer.allowed_solvers}")
```

!!! note
    The `allowed_solvers` shouldn't be changed, changing the default values of this attribute will 
    result in an exception being raised.

Except for `Unregularized`, all the other `Regularizer` objects will have the `regularizer_strength` hyper-parameter.
This is particularly helpful for controlling over-fitting. In general, the larger the `regularizer_strength` 
the smaller the coefficients. Usually, one should tune this hyper-parameter by means of cross-validation. Look at the
[Integration  with `scikit-learn`](#interactions-with-scikit-learn) session for a concrete example.

### Interactions with `pynapple`

!!! warning
    This section assumes some familiarity with the `pynapple` package for time-series manipulation and data 
    exploration. If you'd like to learn more about it, take a look at the [`pynapple` documentation](https://pynapple-org.github.io/pynapple/).

If you represent your task variables and/or spike counts as `pynapple` time-series, don't worry, `nemos` estimators are 
fully compatible with it. You can pass your time series direclty to the `fit` and `score` methods, as well as to 
`predict`.

In `nemos`, when a transformation preserve the time-axis, if a `pynapple` time-series is provided as input, 
the output will be a `pynapple` time-series too!

A canonical example of this behavior is the `predict` method of `GLM`. The method receives a time-series of features as 
input  and outputs a corresponding time-series of firing rates.


```python
import nemos as nmo
import numpy as np
import pynapple as nap

# predictors and observed counts as pynapple time-series with data
time = np.arange(100)
X = nap.TsdTensor(t=time, d=0.2 * np.random.normal(size=(100, 1, 1)))
coef = np.random.normal(size=(1, 1))
# compute the firing rate as a weighted sum of the feature
firing_rate = np.exp(np.einsum("nf, tnf -> tn", coef, X))
# generate spikes
y = nap.TsdFrame(t=time, d=np.random.poisson(firing_rate))

model = nmo.glm.GLM()
# the following works
model.fit(X, y)

# this will return a pynapple TsdFrame
print(type(model.predict(X)))
```

#### Why should you care?

`pynapple` is an extremely helpful tool when working with time series data. You can easily perform operations such 
as restricting your time-series to specific epochs (sleep/wake, context A vs context B. etc.), as well as common 
pre-processing steps in a robust and efficient manner. This includes bin-averaging, counting, convolving, smoothing and many
others. All these operations can be easily concatenated for a quick and easy time-series manipulation.

Combining `pynapple` and `nemos` can greatly streamline the modeling process. To show how handy this could 
be, let's simulate a common scenario for many systems neuro-scientists: we record a behavioral
feature with a temporal resolution, but we want to model spike counts at a different resolution. On top of that,
let's assume that we want to restrict our fit to a specific time epoch.

All of this could be done in a single line of code.

First, let's simulate our feature and the spike times of a neuron, and define the recording epoch and the 
epoch that we want to use for fitting (let's assume it marks when the subject was awake).

```python
import numpy as np
import pynapple as nap

# Assume that this is your input before any processing:
# - recording duration: 5 min 
# - task variable sampled at 10KHz
# - spike times in seconds.

# define time axis and a feature
time_sec = np.linspace(0, 300, 300 * 10000)
X = nap.TsdTensor(t=time_sec, d=0.2 * np.random.normal(size=(time_sec.shape[0], 1, 1)))
# define epochs as pynapple IntervalSet
recording_epoch = nap.IntervalSet(start=0, end=300)
wake_epoch = nap.IntervalSet(start=0, end=150)               
# deine spikes by generating random times, and adding the recording_epoch as the time_support for the spikes
spike_ts = nap.Ts(np.sort(np.random.uniform(0, 300, size=100)), time_support=recording_epoch)
spikes = nap.TsGroup({1: spike_ts})
```

Finally, let's process and fit our data.

```python
import nemos as nmo

# Actual processing and fitting:
# - Restrict to the wake epoch
# - Down-sample to 10ms the features
# - Bin the spike to the same resolution
# - Fit a GLM.
nmo.glm.GLM().fit(X.bin_average(0.01).restrict(wake_epoch), spikes.count(0.01).restrict(wake_epoch))
```

!!! note
    In this example we do all the processing and fitting in a single line to showcase how versatile this approach can
    be. In general, you should always avoid nesting many processing steps without each transformation: what if you
    unintentionally used the wrong bin-size? What if you selected the wrong feature? 

### Interactions with `scikit-learn`

As previously mentioned, `nemos` GLM conforms to the `scikit-learn` API for estimators. As a consequence, 
you can retrieve the all the parameters and set any of them using the `get_param` and `set_param` methods.

```python
import nemos as nmo

model = nmo.glm.GLM()

# get a dictionary of attributes, including nested once.
parameter_dict = model.get_params()
print(parameter_dict)


# set a model attribute
model.set_params(regularizer=nmo.regularizer.Lasso())
print(model.get_params()["regularizer"])

# set a nested attribute
model.set_params(regularizer__regularizer_strength=10)
print(model.get_params())
```

This is used internally by `scikit-learn` to construct complex pipelines.

#### Why should you care?

Respecting the scikit-learn API will allow us to access their powerful pipeline and cross-validation machinery.
All of this, while relying on JAX in the backend for code efficiency (GPU-acceleration).

For example, if we would like to tune the critical hyper-parameter `regularization_strenght`, we
could easily run a `K-Fold` cross-validation using through `scikit-learn`.

```python
import nemos as nmo
import numpy as np
from sklearn.model_selection import GridSearchCV

# predictors, true coefficients, and observed counts
X = 0.2 * np.random.normal(size=(100, 1, 1))
coef = np.random.normal(size=(1, 1))
y = np.random.poisson(np.exp(np.einsum("nf, tnf -> tn", coef, X)))

# model definition
model = nmo.glm.GLM(regularizer=nmo.regularizer.Ridge())

# fit a 5-fold cross-validation scheme
param_grid = dict(regularizer__regularizer_strength=(0.01, 0.001))
cls = GridSearchCV(model, param_grid=param_grid, cv=5)
cls.fit(X, y)

# print best regularizer strength
print(cls.best_params_)
```

