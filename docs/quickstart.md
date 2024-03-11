
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

For more information on how to change default arguments, see the API guide for [`observation_models`]() and
[`regularizer`]().


[//]: # (You can set the defaults when you instantiate a model.)

[//]: # ()
[//]: # (```python)

[//]: # (import nemos as nmo)

[//]: # ()
[//]: # (# set no regularization at initialization)

[//]: # (model = nmo.glm.GLM&#40;regularizer=nmo.regularizer.UnRegularized&#40;&#41;&#41;)

[//]: # (print&#40;model.regularizer&#41;)

[//]: # (```)

[//]: # ()
[//]: # (#### Regularizer Hyperparameters)

[//]: # ()
[//]: # (We specify objective function regularization using regularizer objects. Each have different input arguments )

[//]: # (&#40;see [API documentation]&#40;../reference/nemos/regularizer#Regularizer&#41; for more details&#41;, but all share the following:)

[//]: # ()
[//]: # (1. `solver_name`: The name of the [`jaxopt`]&#40;https://jaxopt.github.io/stable/&#41; solver used for learning the model parameters &#40;for instance, )

[//]: # (    `GradientDescent`, `BFGS`, etc.&#41;)

[//]: # (2. `solver_kwargs`: Additional arguments that should be passed to the solver &#40;for instance, `tol`, `max_iter` and )

[//]: # (    similar&#41; as a dictionary.)

[//]: # ()
[//]: # (For each regularizer, we have one or more allowable solvers, stored in the `allowed_solvers` attribute. )

[//]: # (This ensures that for each regularization scheme, we run an appropriate optimization algorithm. )

[//]: # ()
[//]: # (```python)

[//]: # (import nemos as nmo)

[//]: # ()
[//]: # (regularizer = nmo.regularizer.Ridge&#40;&#41;)

[//]: # (print&#40;f"Allowed solver: {regularizer.allowed_solvers}"&#41;)

[//]: # (```)

[//]: # ()
[//]: # (Except for `Unregularized`, all the other `Regularizer` objects will have the `regularizer_strength` hyper-parameter.)

[//]: # (This is particularly helpful for controlling over-fitting. In general, the larger the `regularizer_strength` )

[//]: # (the smaller the coefficients. Usually, one should tune this hyper-parameter by means of cross-validation. Look at the)

[//]: # ([Integration  with `scikit-learn`]&#40;#interactions-with-scikit-learn&#41; session for a concrete example.)

[//]: # ()
[//]: # (#### Observation Model Input Arguments)

[//]: # ()
[//]: # (The observation model has a single input argument, the non-linearity which maps a linear combination of predictors )

[//]: # (to the neural activity mean &#40;the instantaneous firing rate&#41;. We call the non-linearity *inverse link-function*, )

[//]: # (naming convention from the [statistical literature on  GLMs]&#40;https://en.wikipedia.org/wiki/Generalized_linear_model&#41;.)

[//]: # (The default for the `PoissonObservation` is the exponential $f&#40;x&#41; = e^x$, implemented in JAX as `jax.numpy.exp`. )

[//]: # (Another common choice is the "soft-plus", in JAX this is implemented as `jax.nn.softplus`. )

[//]: # ()
[//]: # (As with all `nemos` objects, one can change the defaults at initialization.)

[//]: # ()
[//]: # (```python)

[//]: # (import jax)

[//]: # (import nemos as nmo)

[//]: # ()
[//]: # (# change default )

[//]: # (obs_model = nmo.observation_models.PoissonObservations&#40;inverse_link_function=jax.nn.softplus&#41;)

[//]: # (model = nmo.glm.GLM&#40;observation_model=obs_model&#41;)

[//]: # (print&#40;model.observation_model.inverse_link_function&#41;)

[//]: # (```)

[//]: # ()
[//]: # (These two options result in a convex optimization objective for all the provided regularizers )

[//]: # (&#40;un-regularized, Ridge, Lasso, group-Lasso&#41;. This is nice because we can guarantee that there exists a single optimal )

[//]: # (set of model coefficients.)

[//]: # ()
[//]: # (!!! info "Can I set my own inverse link function?")

[//]: # (    Yes! You can pass arbitrary python functions to our observation models, provided that jax can differentiate it and )

[//]: # (    it can accept either scalars or arrays as input, returning a scalar or array of the same shape, respectively.)

[//]: # (    However, if you do so, note that we can no longer guarantee the convexity of the optimization procedure! )

### Pre-processing with `pynapple`

!!! warning
    This section assumes some familiarity with the `pynapple` package for time-series manipulation and data 
    exploration. If you'd like to learn more about it, take a look at the [`pynapple` documentation](https://pynapple-org.github.io/pynapple/).

`pynapple` is an extremely helpful tool when working with time series data. You can easily perform operations such 
as restricting your time-series to specific epochs (sleep/wake, context A vs. context B, etc.), as well as common 
pre-processing steps in a robust and efficient manner. This includes bin-averaging, counting, convolving, smoothing and many
others. All these operations can be easily concatenated for a quick and easy data pre-processing.

In `nemos`, if a transformation  preserve the time-axis and you use a `pynapple` time-series as input, the result will 
also be a `pynapple` time-series.

A canonical example of this behavior is the `predict` method of `GLM`. 

```python
>>> print(type(X)) # ...Assume X is a pynapple TsdTensor of shape (num samples, num neurons, num features)
nap.TsdTensor

>>> model.fit(X, y) # the following works

>>> firing_rate = model.predict(X) # predict the firing rate of the neuron

>>> print(type(firing_rate)) # this will still be a pynapple time-series of shape (num_samples, num_neurons)
nap.TsdFrame
```

Let's see how you can greatly streamline your analysis pipeline by integrating `pynapple` and `nemos`

```python
# load head direction data

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

And you can visualize the results,

```python
# tuning raw vs tuning model using pyanapple
tc_model = nap.tuning_curve_continuous_1d(model.predict(X))
tc_raw = nap.tuning_curve_1d(spikes)

plt.plot(tc_raw.index, )
```

!!! note
    In this example we do all the processing and fitting in a single line to showcase how versatile this approach can
    be. In general, you should always avoid nesting many processing steps without each inspecting transformation first: 
    what if you unintentionally used the wrong bin-size? What if you selected the wrong feature? 

### Compatibility with `scikit-learn`

`scikit-learn` is a machine learning toolkit that offers advanced features like pipelines and cross-validation methods. 

In `nemos` takes advantage of these features, while still gaining the benefit of JAX's just-in-time 
compilation and GPU-acceleration!

For example, if we would like to tune the critical hyper-parameter `regularizer_strength`, we
could easily run a `K-Fold` cross-validation using `scikit-learn`.

```python
import nemos as nmo
from sklearn.model_selection import GridSearchCV

# ...Assume X and y are available or generated as shown above

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

