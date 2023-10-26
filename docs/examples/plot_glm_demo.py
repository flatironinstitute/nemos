"""
# GLM Demo: Toy Model Examples

## Introduction

In this demo we will work through two toy example of a Poisson-GLM on synthetic data: a purely feed-forward input model
and a recurrently coupled model.

In particular, we will learn how to:

- Define & configurate a GLM object.
- Fit the model
- Cross-validate the model with `sklearn`
- Simulate spike trains.

Before digging into the GLM module, let's first import the packages
 we are going to use for this tutorial, and generate some synthetic
 data.

"""
import json

import jax
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as sklearn_model_selection
from matplotlib.patches import Rectangle

import neurostatslib as nsl

# Enable float64 precision (optional)
jax.config.update("jax_enable_x64", True)

np.random.seed(111)
# Random design tensor. Shape (n_time_points, n_neurons, n_features).
X = 0.5*np.random.normal(size=(100, 1, 5))

# log-rates & weights, shape (n_neurons, ) and (n_neurons, n_features) respectively.
b_true = np.zeros((1, ))
w_true = np.random.normal(size=(1, 5))

# sparsify weights
w_true[0, 1:4] = 0.

# generate counts
rate = jax.numpy.exp(jax.numpy.einsum("ik,tik->ti", w_true, X) + b_true[None, :])
spikes = np.random.poisson(rate)

# %%
# ## The Feed-Forward GLM
#
# ### Model Definition
# The class implementing the  feed-forward GLM is `neurostatslib.glm.GLM`.
# In order to define the class, one **must** provide:
#
# - **Noise Model**: The noise model for the GLM, e.g. an object of the class of type
# `neurostatslib.noise_model.NoiseModel`. So far, only the `PoissonNoiseModel` noise
# model has been implemented.
# - **Solver**: The desired solver, e.g. an object of the `neurostatslib.solver.Solver` class.
# Currently, we implemented the un-regulrized, Ridge, Lasso, and Group-Lasso solver.
#
# The default for the GLM class is the `PoissonNoiseModel` with log-link function with a Ridge solver.
# Here is how to define the model.

# default Poisson GLM with Ridge solver and Poisson noise model.
model = nsl.glm.GLM()

print("Solver type:     ", type(model.solver))
print("Noise model type:",type(model.noise_model))

# %%
# ### Model Configuration
# One could visualize the model hyperparameters by calling `get_params` method.

# Get the glm model parameters only
print("\nGLM model parameters:")
for key, value in model.get_params(deep=False).items():
    print(f"\t- {key}: {value}")

# Get the glm model parameters, including the all the
# attributes
print("\nNested parameters:")
for key, value in model.get_params(deep=True).items():
    if key in model.get_params(deep=False):
        continue
    print(f"\t- {key}: {value}")

# %%
# These parameters can be configured at initialization and/or
# set after the model is initialized with the following syntax:

# Poisson noise model with soft-plus NL
noise_model = nsl.noise_model.PoissonNoiseModel(jax.nn.softplus)

# Observation noise
solver = nsl.solver.RidgeSolver(
    solver_name="LBFGS",
    regularizer_strength=0.1,
    solver_kwargs={"tol":10**-10}
)

# define the GLM
model = nsl.glm.GLM(
    noise_model=noise_model,
    solver=solver,
)

print("Solver type:     ", type(model.solver))
print("Noise model type:",type(model.noise_model))

# %%
# Hyperparameters can be set at any moment via the `set_params` method.

model.set_params(
    solver=nsl.solver.LassoSolver(),
    noise_model__inverse_link_function=jax.numpy.exp
)

print("Updated solver: ", model.solver)
print("Updated NL: ", model.noise_model.inverse_link_function)

# %%
# !!! warning
#     Each `Solver` has an associated attribute `Solver.allowed_optimizers`
#     which lists the optimizers that are suited for each optimization problem.
#     For example, a RidgeSolver is differentiable and can be fit with `GradientDescent`
#     , `BFGS`, etc., while a LassoSolver should use the `ProximalGradient` method instead.
#     If the provided `solver_name` is not listed in the `allowed_optimizers` this will raise an
#     exception.

# %%
# ### Model Fit
# Fitting the model is as straight forward as calling the `model.fit`
# providing the design tensor and the population counts.
# Additionally one may provide an initial parameter guess.
# The same exact syntax works for any configuration.

# Fit a ridge regression Poisson GLM
model = nsl.glm.GLM()
model.set_params(solver__regularizer_strength=0.1)
model.fit(X, spikes)

print("Ridge results")
print("True weights:      ", w_true)
print("Recovered weights: ", model.basis_coeff_)

# %%
# ## K-fold Cross Validation with `sklearn`
# Our implementation follows the `scikit-learn` api,  this enables us
# to take advantage of the `scikit-learn` tool-box seamlessly, while at the same time
# we take advantage of the `jax` GPU acceleration and auto-differentiation in the
# back-end.
#
# Here is an example of how we can perform 5-fold cross-validation via `scikit-learn`.
# **Ridge**

parameter_grid = {"solver__regularizer_strength": np.logspace(-1.5, 1.5, 6)}
cls = sklearn_model_selection.GridSearchCV(model, parameter_grid, cv=5)
cls.fit(X, spikes)

print("Ridge results        ")
print("Best hyperparameter: ", cls.best_params_)
print("True weights:      ", w_true)
print("Recovered weights: ", cls.best_estimator_.basis_coeff_)

# %%
# We can compare the Ridge cross-validated results with other solvers.
#
# **Lasso**

model.set_params(solver=nsl.solver.LassoSolver())
cls = sklearn_model_selection.GridSearchCV(model, parameter_grid, cv=5)
cls.fit(X, spikes)

print("Lasso results        ")
print("Best hyperparameter: ", cls.best_params_)
print("True weights:      ", w_true)
print("Recovered weights: ", cls.best_estimator_.basis_coeff_)

# %%
# **Group Lasso**

# define groups by masking. Mask size (n_groups, n_features)
mask = np.zeros((2, 5))
mask[0, [0, -1]] = 1
mask[1, 1:-1] = 1

solver = nsl.solver.GroupLassoSolver("ProximalGradient", mask=mask)
model.set_params(solver=solver)
cls = sklearn_model_selection.GridSearchCV(model, parameter_grid, cv=5)
cls.fit(X, spikes)

print("\nGroup Lasso results")
print("Group mask:          :")
print(mask)
print("Best hyperparameter: ", cls.best_params_)
print("True weights:      ", w_true)
print("Recovered weights: ", cls.best_estimator_.basis_coeff_)

# %%
# ## Simulate Spikes
# We can generate spikes in response to a feedforward-stimuli
# through the `model.simulate` method.

# here we are creating a new data input, of 20 timepoints (arbitrary)
# with the same number of neurons and features (mandatory)
Xnew = np.random.normal(size=(20, ) + X.shape[1:])
# generate a random key given a seed
random_key = jax.random.PRNGKey(123)
spikes, rates = model.simulate(random_key, Xnew)

plt.figure()
plt.eventplot(np.where(spikes)[0])


# %%
# ## Recurrently Coupled GLM
# Defining a recurrent model follows the same syntax. In this example
# we will simulate two coupled neurons. and we will inject a transient
# input driving the rate of one of the neurons.
#
# For brevity, we will import the model parameters instead of generating
# them on the fly.

# load parameters
with open("coupled_neurons_params.json", "r") as fh:
    config_dict = json.load(fh)

# basis weights & intercept for the GLM (both coupling and feedforward)
# (the last coefficient is the weight of the feedforward input)
basis_coeff = np.asarray(config_dict["basis_coeff_"])[:, :-1]

# Mask the weights so that only the first neuron receives the imput
basis_coeff[:, 40:] = np.abs(basis_coeff[:, 40:]) * np.array([[1.], [0.]])

baseline_log_fr = np.asarray(config_dict["baseline_link_fr_"])

# basis function, inputs and initial spikes
coupling_basis = jax.numpy.asarray(config_dict["coupling_basis"])
feedforward_input = jax.numpy.asarray(config_dict["feedforward_input"])
init_spikes = jax.numpy.asarray(config_dict["init_spikes"])

# %%
# We can explore visualize the coupling filters and the input.

# plot coupling functions
n_basis_coupling = coupling_basis.shape[1]
fig, axs = plt.subplots(2,2)
plt.suptitle("Coupling filters")
for neu_i in range(2):
    for neu_j in range(2):
        axs[neu_i,neu_j].set_title(f"neu {neu_j} -> neu {neu_i}")
        coeff = basis_coeff[neu_i, neu_j*n_basis_coupling: (neu_j+1)*n_basis_coupling]
        axs[neu_i, neu_j].plot(np.dot(coupling_basis, coeff))
plt.tight_layout()

fig, axs = plt.subplots(1,1)
plt.title("Feedforward inputs")
plt.plot(feedforward_input[:, 0])


# %%
# We can now simulate spikes by calling the `simulate_recurrent` method.

model = nsl.glm.GLMRecurrent()
model.basis_coeff_ = jax.numpy.asarray(basis_coeff)
model.baseline_link_fr_ = jax.numpy.asarray(baseline_log_fr)


# call simulate, with both the recurrent coupling
# and the input
spikes, rates = model.simulate_recurrent(
    random_key,
    feedforward_input=feedforward_input,
    coupling_basis_matrix=coupling_basis,
    init_y=init_spikes
)

# %%
# And finally plot the results for both neurons.

# mkdocs_gallery_thumbnail_number = 4
plt.figure()
ax = plt.subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

patch = Rectangle((200, -0.011), 300, 0.15,  alpha=0.2, color="grey")

p0, = plt.plot(rates[:, 0])
p1, = plt.plot(rates[:, 1])

plt.vlines(np.where(spikes[:, 0])[0], 0.00, 0.01, color=p0.get_color(), label="neu 0")
plt.vlines(np.where(spikes[:, 1])[0], -0.01, 0.00, color=p1.get_color(), label="neu 1")
plt.plot(np.exp(basis_coeff[0, -1] * feedforward_input[:, 0, 0] + baseline_log_fr[0]), color='k', lw=0.8, label="stimulus")
ax.add_patch(patch)
plt.ylim(-0.011, .13)
plt.ylabel("count/bin")
plt.legend()


