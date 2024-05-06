
This tutorial will introduce the main `nemos` functionalities. This is intended for users that are 
already familiar with the GLM framework but want to learn how to interact with the `nemos` API. 
If you have used [scikit-learn](https://scikit-learn.org/stable/) before, we are compatible with the [estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html), so the quickstart should look 
familiar.

In the following sessions, you will learn:

1. [How to define and fit a GLM model.](#basic-model-fitting)
2. [What are the GLM input arguments.](#model-arguments)
3. [How to use `nemos` with `pynapple` for pre-processing.](#pre-processing-with-pynapple)
4. [How to use `nemos` with `scikit-learn` for pipelines and cross-validation.](#compatibility-with-scikit-learn)

Each of these sections can be run independently of the others.

### Basic Model Fitting

Defining and fitting a `nemos` GLM model is straightforward:

```python
import nemos as nmo
import numpy as np

# predictors, shape (n_samples, n_features)
X = 0.2 * np.random.normal(size=(100, 1))
# true coefficients, shape (n_features)
coef = np.random.normal(size=(1, ))
# observed counts, shape (n_samples, )
y = np.random.poisson(np.exp(np.matmul(X, coef)))

# model definition
model = nmo.glm.GLM()
# model fitting
model.fit(X, y)
```


Once fit, you can retrieve model parameters as follows,

```python
>>> # model coefficients, shape (n_features, )
>>> print(f"Model coefficients: {model.coef_}")
Model coefficients: [-1.5791758]

>>> # model coefficients, shape (1, )
>>> print(f"Model intercept: {model.intercept_}")
Model intercept: [-0.0010547]
```


### Model Arguments

During initialization, the `GLM` class accepts the following optional input arguments,

1. `model.observation_model`: The statistical model for the observed variable. The available option so far are `nemos.observation_models.PoissonObservation` and  `nemos.observation_models.GammaObservations`, which are the most common choices for modeling spike counts and calcium imaging traces respectively.
2. `model.regularizer`: Determines the regularization type, defaulting to `nemos.regularizer.Unregularized`.

For more information on how to change default arguments, see the API guide for [`observation_models`](../reference/nemos/observation_models/) and
[`regularizer`](../reference/nemos/regularizer/).

```python
import nemos as nmo

# initialize a Gamma GLM with Ridge regularization
model = nmo.glm.GLM(
    regularizer=nmo.regularizer.Ridge(), 
    observation_model=nmo.observation_models.GammaObservations()
)
```


### Pre-processing with `pynapple`

!!! warning
    This section assumes some familiarity with the `pynapple` package for time series manipulation and data 
    exploration. If you'd like to learn more about it, take a look at the [`pynapple` documentation](https://pynapple-org.github.io/pynapple/).

`pynapple` is an extremely helpful tool when working with time series data. You can easily perform operations such 
as restricting your time series to specific epochs (sleep/wake, context A vs. context B, etc.), as well as common 
pre-processing steps in a robust and efficient manner. This includes bin-averaging, counting, convolving, smoothing and many
others. All these operations can be easily concatenated for a quick and easy data pre-processing.

In `nemos`, if a transformation  preserve the time axis and you use a `pynapple` time series as input, the result will 
also be a `pynapple` time series.

A canonical example of this behavior is the `predict` method of `GLM`.

```python
>>>  # Assume X is a pynapple TsdFrame
>>> print(type(X))  # shape (num samples, num features)
<class 'pynapple.core.time_series.TsdFrame'>

>>> model.fit(X, y)  # the following works

>>> firing_rate = model.predict(X)  # predict the firing rate of the neuron

>>>  # this will still be a pynapple time series
>>> print(type(firing_rate))  # shape (num_samples, )
<class 'pynapple.core.time_series.Tsd'>
```

Let's see how you can greatly streamline your analysis pipeline by integrating `pynapple` and `nemos`.

!!! note
    You can download this dataset by clicking [here](https://www.dropbox.com/s/su4oaje57g3kit9/A2929-200711.zip?dl=1).

```python
import nemos as nmo
import numpy as np
import pynapple as nap

data = nap.load_file("A2929-200711.nwb")

spikes = data["units"]
head_dir = data["ry"]

counts = spikes[6].count(0.01, ep=head_dir.time_support)  # restrict and bin
upsampled_head_dir = head_dir.bin_average(0.01)  # up-sample head direction

# create your features
X = nmo.basis.CyclicBSplineBasis(10).compute_features(upsampled_head_dir)

# add a neuron axis and fit model
model = nmo.glm.GLM().fit(X, counts) 
```

Finally, let's compare the tuning curves

```python
import numpy as np
import matplotlib.pyplot as plt

raw_tuning = nap.compute_1d_tuning_curves(spikes, head_dir, nb_bins=100)[6]
model_tuning = nap.compute_1d_tuning_curves_continuous(
    model.predict(X)[:, np.newaxis] * X.rate,  # scale by the sampling rate
    head_dir,
    nb_bins=100
)[0]

# plot results
plt.subplot(111, projection="polar")
plt.plot(raw_tuning.index, raw_tuning.values, label="raw")
plt.plot(model_tuning.index, model_tuning.values, label="glm")
plt.legend()
plt.yticks([])
plt.xlabel("heading angle")
plt.show()
```

![Alt text](head_dir_tuning.jpg)

### Compatibility with `scikit-learn`

`scikit-learn` is a machine learning toolkit that offers advanced features like pipelines and cross-validation methods. 

`nemos` takes advantage of these features, while still gaining the benefit of JAX's just-in-time 
compilation and GPU-acceleration!

For example, if we would like to tune the critical hyper-parameter `regularizer_strength`, we
could easily run a `K-Fold` cross-validation using `scikit-learn`.

```python
import nemos as nmo
from sklearn.model_selection import GridSearchCV

# ...Assume X and counts are available or generated as shown above

# model definition
model = nmo.glm.GLM(regularizer=nmo.regularizer.Ridge())

# fit a 5-fold cross-validation scheme for comparing two different
# regularizer strengths:

# - define the parameter grid
param_grid = dict(regularizer__regularizer_strength=(0.01, 0.001))

# - define the 5-fold cross-validation grid search from sklearn
cls = GridSearchCV(model, param_grid=param_grid, cv=5)

# - run the 5-fold cross-validation grid search
cls.fit(X, counts)
```

Now we can print the best coefficient.

```python
# print best regularizer strength
>>> print(cls.best_params_)
{'regularizer__regularizer_strength': 0.001}
```

Enjoy modeling with `nemos`!
