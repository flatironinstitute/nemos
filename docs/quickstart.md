---
hide:
  - navigation
---
## __Overview__

NeMoS is a neural modeling software package designed to model neural spiking activity and other time-series data. 

At its core, NeMoS consists of two primary modules: the __`basis`__ and the __`glm`__ module.

The `basis` module focuses on designing model features (inputs) for the GLM. It includes a suite of composable feature 
constructors that accept time-series data as inputs. These inputs can be any observed variables, such as presented 
stimuli, head direction, position, or spike counts. 

The basis objects can perform two types of transformations on the inputs:

1. **Non-linear Mapping:** This process transforms the input data through a non-linear function, 
   allowing it to capture complex, non-linear relationships between inputs and neuronal firing rates. 
   Importantly, this transformation preserves the properties that makes GLM easy to fit and guarantee a 
   single optimal solution (e.g. convexity).

2. **Convolution:** This applies a convolution of the input data with a bank of filters, designed to 
   capture linear temporal effects. This transformation is particularly useful when analyzing data with 
   inherent time dependencies or when the temporal dynamics of the input are significant.

Both transformations produce a vector of features `X` that changes over time, with a shape 
of `(n_time_points, n_features)`.

On the other hand, the `glm` module maps the feature to spike counts. It is used to learn the GLM weights, 
evaluating the model performance, and explore its behavior on new input.

## __Basic Usage__

Here's a brief demonstration of how the `basis` and `glm` modules work together within NeMoS.

### Poisson GLM for Feature Analysis

!!! info "Non-Linear Mapping"
    This section showcases the use of basis as a non-linear mapping.<br>
    For a real-data example, check out [this tutorial](../generated/tutorials/plot_05_place_cells).

<img src="../assets/glm_features_scheme.svg" width="100%">

In this example, we'll construct a time-series of features using the basis objects.


##### Feature Representation

```python
import nemos as nmo
import numpy as np

# define 3 input features of shape (n_samples, )
input_1, input_2, input_3 = 0.2 * np.random.normal(size=(3, 100))

# Instantiate the basis
basis_1 = nmo.basis.MSplineBasis(n_basis_funcs=5)
basis_2 = nmo.basis.CyclicBSplineBasis(n_basis_funcs=6)
basis_3 = nmo.basis.MSplineBasis(n_basis_funcs=7)

# Combine the basis functions with addition and multiplication
basis = basis_1 * basis_2 + basis_3

# Generate the design matrix starting from some raw 
# input time series, i.e. LFP phase, position, etc.
X = basis.compute_features(input_1, input_2, input_3)
```

##### Fit a GLM

```python

# assume that we observe these spike_counts of shape (n_samples, )
spike_counts = np.random.poisson(size=X.shape[0])

# model definition
model = nmo.glm.GLM()
# model fitting
model.fit(X, spike_counts)
```

Once the model is fit, you can retrieve the model parameters as shown below.

```python
>>> # model coefficients shape is (37, ), or 5 * 6 + 7
>>> print(f"Model coefficients shape: {model.coef_.shape}")
Model coefficients shape: (37, )

>>> # model coefficients, shape (1, )
>>> print(f"Model intercept: {model.intercept_}")
Model intercept: [-0.26412687]
```

### Poisson GLM for Neural Population

!!! info "Convolution"
    This section demonstrates how to use the basis module to convolve an input (in this case, spike counts) 
    with a bank of kernels.<br>
    For a real-data example, check out [this tutorial](../generated/tutorials/plot_02_head_direction).

<img src="../assets/glm_population_scheme.svg" width="84%">

This second example demonstrates feature construction by convolving the simultaneously recorded population spike counts with a bank of filters, utilizing the basis in `conv` mode.
The figure above show the GLM scheme for a single neuron, however in NeMoS you can fit jointly the whole population with the [`PopulationGLM`](generated/how_to_guide/plot_04_population_glm.md) object.

#### Feature Representation

```python
# assume that the population spike counts time-series is stored 
# in a 2D array spike_counts of shape (n_samples, n_neurons).
spike_counts = np.random.poisson(size=(1000, 3))  

# generate 5 basis functions of 100 time-bins, 
# and convolve the counts with the basis.
X = nmo.basis.RaisedCosineBasisLog(5, mode="conv", window_size=100
    ).compute_features(spike_counts)
```

!!! note "Multi-epoch convolution"
    If your data (such as spike_counts in this example) is formatted as a [`pynapple` time-series](#pre-processing-with-pynapple), 
    the convolution performed by the basis objects will be done epoch-by-epoch, helping to avoid artifacts 
    introduced by gaps in your time-series.

#### Fit a Population GLM

```python
# fit a GLM to the first neuron counts time-series
glm = nmo.glm.PopulationGLM().fit(X, spike_counts)

# compute the rate
firing_rate = glm.predict(X)

# compute log-likelihood
ll = glm.score(X, spike_counts)
```

## __Continuous Observations__

By default, NeMoS' GLM uses Poisson observations, which are a natural choice for spike counts. However, the package also supports a Gamma GLM, which is more appropriate for modeling continuous, non-negative observations such as calcium transients.

To change the default observation model, follow the example below,

```python
import nemos as nmo

# set up a Gamma GLM for modeling continuous non-negative data
glm = nmo.glm.GLM(observation_model=nmo.observation_models.GammaObservations())
```

Take a look at our [tutorial](../generated/tutorials/plot_06_calcium_imaging) for a detailed example.

## __Regularization__

NeMoS supports various regularization schemes, including Ridge ($L_2$), Lasso ($L_1$), and Group Lasso, to prevent overfitting and improve model generalization.

You can specify the regularization scheme and its strength when initializing the GLM model, as shown below:

```python
import nemos as nmo

# Instantiate a GLM with Ridge (L2) regularization
glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.1)
```


## __Pre-processing with `pynapple`__

!!! warning
    This section assumes some familiarity with the `pynapple` package for time series manipulation and data 
    exploration. If you'd like to learn more about it, take a look at the [`pynapple` documentation](https://pynapple-org.github.io/pynapple/).

`pynapple` is an extremely helpful tool when working with time series data. You can easily perform operations such 
as restricting your time series to specific epochs (sleep/wake, context A vs. context B, etc.), as well as common 
pre-processing steps in a robust and efficient manner. This includes bin-averaging, counting, convolving, smoothing and many
others. All these operations can be easily concatenated for a quick and easy data pre-processing.

In NeMoS, if a transformation  preserve the time axis and you use a `pynapple` time series as input, the result will 
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

Let's see how you can greatly streamline your analysis pipeline by integrating `pynapple` and NeMoS.

!!! note
    You can download this dataset by clicking [here](https://www.dropbox.com/s/su4oaje57g3kit9/A2929-200711.zip?dl=1).

```python
import nemos as nmo
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

## __Compatibility with `scikit-learn`__

`scikit-learn` is a machine learning toolkit that offers advanced features like pipelines and cross-validation methods. 

NeMoS takes advantage of these features, while still gaining the benefit of JAX's just-in-time 
compilation and GPU-acceleration!

For example, if we would like to tune the critical hyper-parameter `regularizer_strength`, we
could easily run a `K-Fold` cross-validation using `scikit-learn`.

```python
import nemos as nmo
from sklearn.model_selection import GridSearchCV

# ...Assume X and counts are available or generated as shown above

# model definition
model = nmo.glm.GLM(regularizer="Ridge")

# fit a 5-fold cross-validation scheme for comparing two different
# regularizer strengths:

# - define the parameter grid
param_grid = dict(regularizer__regularizer_strength=(0.01, 0.001))

# - define the 5-fold cross-validation grid search from sklearn
cls = GridSearchCV(model, param_grid=param_grid, cv=5)

# - run the 5-fold cross-validation grid search
cls.fit(X, counts)
```

!!! info "Cross-Validation"
    For more information and a practical example on how to construct a parameter grid and cross-validate hyperparameters across an entire pipeline, please refer to the [tutorial on pipelining and cross-validation](../generated/how_to_guide/plot_06_sklearn_pipeline_cv_demo).

Now we can print the best coefficient.

```python
# print best regularizer strength
>>> print(cls.best_params_)
{'regularizer__regularizer_strength': 0.001}
```

Enjoy modeling with NeMoS!
