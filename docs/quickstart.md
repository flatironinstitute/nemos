
This tutorial will introduce the main `nemos` functionalities. This is intended for users that are 
already familiar with the GLM framework but want to learn how to interact with the `nemos` API. 
If you have used [scikit-learn](https://scikit-learn.org/stable/) before, we are compatible with the [estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html), so the quickstart should look 
familiar.

In the following sessions, you will learn:

1. [How to define and fit a GLM model.](#basic-model-fitting)
2. [How to tweak the GLM using the input arguments.](#model-arguments)
3. [How to use `nemos` with `pynapple` for pre-processing.](#pynapple-compatibility)
4. [How to use `nemos` with `scikit-learn` for pipelines and cross-validation.](#scikit-learn-compatibility)

Each of these sections can be run independently of the others.

### Basic Model Fitting

Defining and fitting a `nemos` GLM model is straightforward:

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
print(f"Model coefficients: {model.coef_}")

# model coefficients, shape (n_neurons, )
print(f"Model intercept: {model.intercept_}")
```

!!! note
    This API is the same as scikit-learn's, see for example their [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

### Model Arguments

During initialization, the `GLM` class accepts the following optional input arguments,

1. `model.observation_model`: The statistical model for the observed variable. The only available option so far is `nemos.observation_models.PoissonObservation`, which is the most common choice for modeling spike counts.
2. `model.regularizer`: Determines the regularization type, defaulting to `nemos.regularizer.Ridge`, for $L_2$ regularization.

You can set the defaults when you instantiate a model.

```python
import nemos as nmo

# set no regularization at initialization
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized())
print(model.regularizer)
```

#### Regularizer Hyperparameters

We specify objective function regularization using regularizer objects. Each have different input arguments 
(see [API documentation](../reference/nemos/regularizer#Regularizer) for more details), but all share the following:

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

Except for `Unregularized`, all the other `Regularizer` objects will have the `regularizer_strength` hyper-parameter.
This is particularly helpful for controlling over-fitting. In general, the larger the `regularizer_strength` 
the smaller the coefficients. Usually, one should tune this hyper-parameter by means of cross-validation. Look at the
[Integration  with `scikit-learn`](#interactions-with-scikit-learn) session for a concrete example.

#### Observation Model Input Arguments

The observation model has a single input argument, the non-linearity which maps a linear combination of predictors 
to the neural activity mean (the instantaneous firing rate). We call the non-linearity *inverse link-function*, 
naming convention from the [statistical literature on  GLMs](https://en.wikipedia.org/wiki/Generalized_linear_model).
The default for the `PoissonObservation` is the exponential $f(x) = e^x$, implemented in JAX as `jax.numpy.exp`. 
Another common choice is the "soft-plus", in JAX this is implemented as `jax.nn.softplus`. 

As with all `nemos` objects, one can change the defaults at initialization.

```python
import jax
import nemos as nmo

# change default 
obs_model = nmo.observation_models.PoissonObservations(inverse_link_function=jax.nn.softplus)
model = nmo.glm.GLM(observation_model=obs_model)
print(model.observation_model.inverse_link_function)
```

These two options result in a convex optimization objective for all the provided regularizers 
(un-regularized, Ridge, Lasso, group-Lasso). This is nice because we can guarantee that there exists a single optimal 
set of model coefficients.

!!! info "Can I set my own inverse link function?"
    Yes! You can pass arbitrary python functions to our observation models, provided that jax can differentiate it and 
    it can accept either scalars or arrays as input, returning a scalar or array of the same shape, respectively.
    However, if you do so, note that we can no longer guarantee the convexity of the optimization procedure! 

### `pynapple` compatibility

!!! warning
    This section assumes some familiarity with the `pynapple` package for time-series manipulation and data 
    exploration. If you'd like to learn more about it, take a look at the [`pynapple` documentation](https://pynapple-org.github.io/pynapple/).

If you represent your task variables and/or spike counts as `pynapple` time-series, don't worry, `nemos` estimators are 
fully compatible with it. You can pass your time series directly to any of our functions.

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

# predict the firing rate of the neuron
firing_rate = model.predict(X)
# this will still be a pynapple time-seris
print(type(firing_rate))
```

#### Why should you care?

`pynapple` is an extremely helpful tool when working with time series data. You can easily perform operations such 
as restricting your time-series to specific epochs (sleep/wake, context A vs. context B, etc.), as well as common 
pre-processing steps in a robust and efficient manner. This includes bin-averaging, counting, convolving, smoothing and many
others. All these operations can be easily concatenated for a quick and easy time-series manipulation.

Combining `pynapple` and `nemos` can greatly streamline the modeling process. To show how handy this could 
be, let's simulate a common scenario for many systems neuro-scientists: we record a behavioral
feature with a temporal resolution, but we want to model spike counts at a different resolution. On top of that,
let's assume that we want to restrict our fit to a specific time epoch.

All of this could be done with a single command.

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

# deine spikes by generating random times, and 
# adding the recording_epoch as the time_support
spike_ts = nap.Ts(
    np.sort(np.random.uniform(0, 300, size=100)), 
    time_support=recording_epoch
)
spikes = nap.TsGroup({1: spike_ts})
```

Finally, let's process and fit our data.

```python
import nemos as nmo

# Actual processing and fitting:

# - Restrict to the wake epoch
X_wake = X.restrict(wake_epoch)
spikes_wake = spikes.restrict(wake_epoch)

# - Down-sample to 10ms the features
X_downsample = X_wake.bin_average(0.01)

# - Bin the spike to the same resolution
counts = spikes_wake.count(0.01)

# - Fit a GLM.
nmo.glm.GLM().fit(X_downsample, counts)
```

Or alternative, in a single command,

```python
nmo.glm.GLM().fit(
    X.bin_average(0.01).restrict(wake_epoch), 
    spikes.count(0.01).restrict(wake_epoch)
)
```

!!! note
    In this example we do all the processing and fitting in a single line to showcase how versatile this approach can
    be. In general, you should always avoid nesting many processing steps without each inspecting transformation first: 
    what if you unintentionally used the wrong bin-size? What if you selected the wrong feature? 

### `scikit-learn` compatibility

As previously mentioned, `nemos` GLM conforms with `scikit-learn` API for estimators.

#### Why should you care?

Respecting the scikit-learn API allows us to make use of their powerful pipeline and cross-validation machinery, 
while still gaining the benefit of JAX's just-in-time compilation and GPU-acceleration!

For example, if we would like to tune the critical hyper-parameter `regularizer_strength`, we
could easily run a `K-Fold` cross-validation using `scikit-learn`.

```python
import nemos as nmo
from sklearn.model_selection import GridSearchCV

# ...Assume X and y are generated as previously shown

# model definition
model = nmo.glm.GLM(regularizer=nmo.regularizer.Ridge())

# fit a 5-fold cross-validation scheme for comparing two different
# regularizer strengths:

# - define the parameter grid
param_grid = dict(regularizer__regularizer_strength=(0.01, 0.001))

# - define the 5-fold cross-validation grid search from sklearn
cls = GridSearchCV(model, param_grid=param_grid, cv=5)

# - run the 5-fold cross-validation grid search
cls.fit(X, y)

# print best regularizer strength
print(cls.best_params_)
```

