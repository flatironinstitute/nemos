"""
# Selecting basis by cross-validation with scikit-learn

In this demo, we will demonstrate how to select an appropriate basis and its hyperparameters using cross-validation.
In particular, we will learn:

1. What a scikit-learn pipeline is.
2. Why pipelines are useful.
3. How to combine NeMoS `Basis` and `GLM` objects in a pipeline.
4. How to select the number of bases and the basis type through cross-validation (or any other hyperparameter in the pipeline).
5. How to use a custom scoring metric to quantify the performance of each configuration.

"""

# %%
# ## What is a scikit-learn pipeline
#
# A pipeline is a sequence of data transformations. Each step in the pipeline transforms the input data into a different representation, with the final step being either another transformation or a model step that fits, predicts, or scores based on the previous step's output and some observations. Setting up such machinery can be simplified using the `Pipeline` class from scikit-learn.
#
# To set up a scikit-learn `Pipeline`, ensure that:
#
# 1. Each intermediate step is a [scikit-learn transformer object](https://scikit-learn.org/stable/data_transforms.html) with a `transform` and/or `fit_transform` method.
# 2. The final step is either another transformer or an [estimator object](https://scikit-learn.org/stable/developers/develop.html#estimators) with a `fit` method, or a model with `fit`, `predict`, and `score` methods.
#
# Each transformation step takes a 2D array `X` of shape `(num_samples, num_original_features)` as input and outputs another 2D array of shape `(num_samples, num_transformed_features)`. The final step takes a pair `(X, y)`, where `X` is as before, and `y` is a 1D array of shape `(n_samples,)` containing the observations to be modeled.
#
# You can define a pipeline as follows:
# ```python
# from sklearn.pipeline import Pipeline
#
# # Assume transformer_i/predictor is a transformer/model object
# pipe = Pipeline(
#     [
#         ("label_1", transformer_1),
#         ("label_2", transformer_2),
#         ...,
#         ("label_n", transformer_n),
#         ("label_model", model)
#     ]
# )
# ```
#
# Calling `pipe.fit(X, y)` will perform the following computations:
# ```python
# # Chain of transformations
# X1 = transformer_1.transform(X)
# X2 = transformer_2.transform(X1)
# # ...
# Xn = transformer_n.transform(Xn_1)
#
# # Fit step
# model.fit(Xn, y)
# ```
# And the same holds for `pipe.score` and `pipe.predict`.
#
# ## Why pipelines are useful
#
# Pipelines not only streamline and simplify your code but also offer several other advantages. The real power of pipelines becomes evident when combined with the scikit-learn `model_selection` module. This combination allows you to tune hyperparameters at each step of the pipeline in a straightforward manner.
#
# In the following sections, we will showcase this approach with a concrete example: selecting the appropriate basis type and number of bases for a GLM regression in NeMoS.
#
# ## Combining basis transformations and GLM in a pipeline
# Let's start by creating some toy data.

import nemos as nmo
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# predictors, shape (n_samples, n_features)
X = np.random.uniform(low=0, high=1, size=(1000, 1))
# observed counts, shape (n_samples,)
rate = 2 * (
    scipy.stats.norm.pdf(X, scale=0.1, loc=0.25)
    + scipy.stats.norm.pdf(X, scale=0.1, loc=0.75)
)
y = np.random.poisson(rate).astype(float).flatten()

# %%
fig, ax = plt.subplots()
ax.scatter(X.flatten(), y, alpha=0.2)
ax.set_xlabel("input")
ax.set_ylabel("spike count")
sns.despine(ax=ax)

# %%
# ### Converting NeMoS `Basis` to a transformer
# In order to use NeMoS `Basis` in a pipeline, we need to convert it into a scikit-learn transformer. This can be achieved through the `TransformerBasis` wrapper class.
#
# Instantiating a `TransformerBasis` can be done using the constructor directly or with `Basis.to_transformer()`:

# %%
bas = nmo.basis.RaisedCosineBasisLinear(5, mode="conv", window_size=5)
trans_bas_a = nmo.basis.TransformerBasis(bas)
trans_bas_b = bas.to_transformer()

# %%
# `TransformerBasis` provides convenient access to the underlying `Basis` object's attributes:

# %%
print(bas.n_basis_funcs, trans_bas_a.n_basis_funcs, trans_bas_b.n_basis_funcs)

# %%
# We can also set attributes of the underlying `Basis`. Note that -- because `TransformerBasis` is created with a copy of the `Basis` object passed to it -- this does not change the original `Basis`, and neither does changing the original `Basis` change `TransformerBasis` we created:

# %%
trans_bas_a.n_basis_funcs = 10
bas.n_basis_funcs = 100

print(bas.n_basis_funcs, trans_bas_a.n_basis_funcs, trans_bas_b.n_basis_funcs)

# %%
# ### Creating and fitting a pipeline
# We might want to combine first transforming the input data with our basis functions, then fitting a GLM on the transformed data.
#
# This is exactly what `Pipeline` is for!

# %%
pipeline = Pipeline(
    [
        (
            "transformerbasis",
            nmo.basis.TransformerBasis(nmo.basis.RaisedCosineBasisLinear(6)),
        ),
        (
            "glm",
            nmo.glm.GLM(regularizer_strength=0.5, regularizer="Ridge"),
        ),
    ]
)

pipeline.fit(X, y)

# %%
# Note how NeMoS models are already scikit-learn compatible and can be used directly in the pipeline.
#
# Visualize the fit:

# %%
fig, ax = plt.subplots()

ax.scatter(X.flatten(), y, alpha=0.2, label="generated spike counts")
ax.set_xlabel("input")
ax.set_ylabel("spike count")

x = np.sort(X.flatten())
ax.plot(
    x,
    pipeline.predict(np.expand_dims(x, 1)),
    label="predicted rate",
    color="tab:orange",
)

ax.legend()
sns.despine(ax=ax)

# %%
# There is some room for improvement, so in the next section we'll use cross-validation to tune the parameters of our pipeline.

# %%
# ### Select the number of basis by cross-validation

# %%
# !!! warning
#     Please keep in mind that while `GLM.score` supports different ways of evaluating goodness-of-fit through the `score_type` argument, `pipeline.score(X, y, score_type="...")` does not propagate this, and uses the default value of `log-likelihood`.
#
#     To evaluate a pipeline, please create a custom scorer (e.g. `pseudo_r2` below) and call `my_custom_scorer(pipeline, X, y)`.
#
# #### Define the parameter grid
#
# Let's define candidate values for the parameters of each step of the pipeline we want to cross-validate. In this case the number of basis functions in the transformation step and the ridge regularization's strength in the GLM fit:

# %%
param_grid = dict(
    glm__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
    transformerbasis__n_basis_funcs=(3, 5, 10, 20, 100),
)

# %%
# #### Run the grid search
# Let's run a 5-fold cross-validation of the hyperparameters with the scikit-learn `model_selection.GridsearchCV` class.

# %%
gridsearch = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5
)

# run the 5-fold cross-validation grid search
gridsearch.fit(X, y)

# %%
# #### Visualize the scores
#
# Let's extract the scores from `gridsearch` and take a look at how the different parameter values of our pipeline influence the test score:

# %%
from matplotlib.patches import Rectangle

def highlight_max_cell(cvdf_wide, ax):
    max_col = cvdf_wide.max().idxmax()
    max_col_index = cvdf_wide.columns.get_loc(max_col)
    max_row = cvdf_wide[max_col].idxmax()
    max_row_index = cvdf_wide.index.get_loc(max_row)

    ax.add_patch(
        Rectangle(
            (max_col_index, max_row_index), 1, 1, fill=False, lw=3, color="skyblue"
        )
    )

# %%
cvdf = pd.DataFrame(gridsearch.cv_results_)

cvdf_wide = cvdf.pivot(
    index="param_transformerbasis__n_basis_funcs",
    columns="param_glm__regularizer_strength",
    values="mean_test_score",
)

ax = sns.heatmap(
    cvdf_wide,
    annot=True,
    square=True,
    linecolor="white",
    linewidth=0.5,
)

# Labeling the colorbar
colorbar = ax.collections[0].colorbar
colorbar.set_label('log-likelihood')

ax.set_xlabel("ridge regularization strength")
ax.set_ylabel("number of basis functions")

highlight_max_cell(cvdf_wide, ax)

# %%
# #### Visualize the predicted
# Finally, visualize the predicted firing rates using the best model found by our grid-search, which gives a better fit than the randomly chosen parameter values we tried in the beginning:

# %%
fig, ax = plt.subplots()

ax.scatter(X.flatten(), y, alpha=0.2, label="generated spike counts")
ax.set_xlabel("input")
ax.set_ylabel("spike count")

x = np.sort(X.flatten())
ax.plot(
    x,
    gridsearch.best_estimator_.predict(np.expand_dims(x, 1)),
    label="predicted rate",
    color="tab:orange",
)

ax.legend()
sns.despine(ax=ax)

# %%
# ### Evaluating different bases directly
#
# In the previous example we set the number of basis functions of the `Basis` wrapped in our `TransformerBasis`. However, if we are for example not sure about the type of basis functions we want to use, or we have already defined some basis functions of our own, then we can use cross-validation to directly evaluate those as well.
#
# Here we include `transformerbasis___basis` in the parameter grid to try different values for `TransformerBasis._basis`:

# %%
param_grid = dict(
    glm__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
    transformerbasis___basis=(
        nmo.basis.RaisedCosineBasisLinear(5),
        nmo.basis.RaisedCosineBasisLinear(10),
        nmo.basis.RaisedCosineBasisLog(5),
        nmo.basis.RaisedCosineBasisLog(10),
        nmo.basis.MSplineBasis(5),
        nmo.basis.MSplineBasis(10),
    ),
)

# %%
# Then run the grid search:

# %%
gridsearch = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
)

# run the 5-fold cross-validation grid search
gridsearch.fit(X, y)


# %%
# Wrangling the output data a bit and looking at the scores:

# %%
cvdf = pd.DataFrame(gridsearch.cv_results_)

# Read out the number of basis functions
cvdf["transformerbasis_config"] = [
    f"{b.__class__.__name__} - {b.n_basis_funcs}"
    for b in cvdf["param_transformerbasis___basis"]
]

cvdf_wide = cvdf.pivot(
    index="transformerbasis_config",
    columns="param_glm__regularizer_strength",
    values="mean_test_score",
)

ax = sns.heatmap(
    cvdf_wide,
    annot=True,
    square=True,
    linecolor="white",
    linewidth=0.5,
)

# Labeling the colorbar
colorbar = ax.collections[0].colorbar
colorbar.set_label('log-likelihood')

ax.set_xlabel("ridge regularization strength")
ax.set_ylabel("number of basis functions")

highlight_max_cell(cvdf_wide, ax)

# %%
# Looks like `RaisedCosineBasisLinear` was probably a decent choice for our toy data:

# %%
fig, ax = plt.subplots()

ax.scatter(X.flatten(), y, alpha=0.2, label="generated spike counts")
ax.set_xlabel("input")
ax.set_ylabel("spike count")

x = np.sort(X.flatten())
ax.plot(
    x,
    gridsearch.best_estimator_.predict(np.expand_dims(x, 1)),
    label="predicted rate",
    color="tab:orange",
)

ax.legend()
sns.despine(ax=ax)

# %%
# !!! warning
#     Please note that because it would lead to unexpected behavior, mixing the two ways of defining values for the parameter grid is not allowed. The following would lead to an error:
#
#     ```python
#     param_grid = dict(
#         glm__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
#         transformerbasis__n_basis_funcs=(3, 5, 10, 20, 100),
#         transformerbasis___basis=(
#             nmo.basis.RaisedCosineBasisLinear(5),
#             nmo.basis.RaisedCosineBasisLinear(10),
#             nmo.basis.RaisedCosineBasisLog(5),
#             nmo.basis.RaisedCosineBasisLog(10),
#             nmo.basis.MSplineBasis(5),
#             nmo.basis.MSplineBasis(10),
#         ),
#     )
#     ```

# %%
# ## Create a custom scorer
# By default, the GLM score method returns the model log-likelihood. If you want to try a different metric, such as the pseudo-R2, you can create a custom scorer that overwrites the default:

# %%
from sklearn.metrics import make_scorer

# NOTE: the order of the arguments is reversed
pseudo_r2 = make_scorer(
    lambda y_true, y_pred: nmo.observation_models.PoissonObservations().pseudo_r2(
        y_pred, y_true
    )
)

# %%
# #### Run the grid search providing the custom scorer

# %%
gridsearch = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=pseudo_r2,
)

# Run the 5-fold cross-validation grid search
gridsearch.fit(X, y)

# %%
# #### Plot the pseudo-R2 scores

# %%
# Plot the pseudo-R2 scores
cvdf = pd.DataFrame(gridsearch.cv_results_)

# Read out the number of basis functions
cvdf["transformerbasis_config"] = [
    f"{b.__class__.__name__} - {b.n_basis_funcs}"
    for b in cvdf["param_transformerbasis___basis"]
]

cvdf_wide = cvdf.pivot(
    index="transformerbasis_config",
    columns="param_glm__regularizer_strength",
    values="mean_test_score",
)

ax = sns.heatmap(
    cvdf_wide,
    annot=True,
    square=True,
    linecolor="white",
    linewidth=0.5,
)

# Labeling the colorbar
colorbar = ax.collections[0].colorbar
colorbar.set_label('pseudo-R2')

ax.set_xlabel("ridge regularization strength")
ax.set_ylabel("number of basis functions")

highlight_max_cell(cvdf_wide, ax)
