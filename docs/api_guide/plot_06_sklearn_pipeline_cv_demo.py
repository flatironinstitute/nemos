"""
# Pipelining and cross-validation with scikit-learn

In this demo we will show how to combine basis transformations and GLMs using scikit-learn's pipelines, and demonstrate how cross-validation can be used to fine-tune parameters of each step of the pipeline.
"""

# %% [markdown]
# ## Let's start by creating some toy data

# %%
import nemos as nmo
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# %%

# predictors, shape (n_samples, n_features)
X = np.random.uniform(low=0, high=1, size=(1000, 1))
# observed counts, shape (n_samples, )
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

# %% [markdown]
# ## Combining basis transformations and GLM in a pipeline

# %% [markdown]
# The `TransformerBasis` wrapper class provides an API consistent with [scikit-learn's data transforms](https://scikit-learn.org/stable/data_transforms.html), allowing its use in pipelines as a transformation step.
#
# Instantiating a `TransformerBasis` can be done using the constructor directly or with `Basis.to_transformer()`:

# %%
bas = nmo.basis.RaisedCosineBasisLinear(5, mode="conv", window_size=5)
trans_bas_a = nmo.basis.TransformerBasis(bas)
trans_bas_b = bas.to_transformer()

# %% [markdown]
# `TransformerBasis` provides convenient access to the underlying `Basis` object's attributes:

# %%
print(bas.n_basis_funcs, trans_bas_a.n_basis_funcs, trans_bas_b.n_basis_funcs)

# %% [markdown]
# We can also set attributes of the underlying `Basis`. Note that -- because `TransformerBasis` is created with a copy of the `Basis` object passed to it -- this does not change the original `Basis`, and neither does changing the original `Basis` change `TransformerBasis` we created:

# %%
trans_bas_a.n_basis_funcs = 10
bas.n_basis_funcs = 100

print(bas.n_basis_funcs, trans_bas_a.n_basis_funcs, trans_bas_b.n_basis_funcs)

# %% [markdown]
# ### Creating and fitting a pipeline
# We might want to combine first transforming the input data with our basis functions, then fitting a GLM on the transformed data.
#
# This is exactly what [scikit-learn's Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline) is for!

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

# %% [markdown]
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

# %% [markdown]
# There is some room for improvement, so in the next section we'll use cross-validation to tune the parameters of our pipeline.

# %% [markdown]
# ## Hyperparameter-tuning with scikit-learn's gridsearch

# %% [markdown]
# !!! warning
#     Please keep in mind that while `GLM.score` supports different ways of evaluating goodness-of-fit through the `score_type` argument, `pipeline.score(X, y, score_type="...")` does not propagate this, and uses the default value of `log-likelihood`.
#
#     To evaluate a pipeline, please create a custom scorer (e.g. `pseudo_r2` below) and call `my_custom_scorer(pipeline, X, y)`.

# %% [markdown]
# ### Evaluating different values of the number of basis functions

# %% [markdown]
# #### Define the parameter grid
#
# Let's define candidate values for the parameters of each step of the pipeline we want to cross-validate. In this case the number of basis functions in the transformation step and the ridge regularization's strength in the GLM fit:

# %%
param_grid = dict(
    glm__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
    transformerbasis__n_basis_funcs=(3, 5, 10, 20, 100),
)

# %% [markdown]
# #### Create a custom scorer
# Create a custom scorer to evaluate models using the pseudo-R2 score instead of the log-likelihood which is the default in `GLM.score`:

# %%
from sklearn.metrics import make_scorer

# NOTE: the order of the arguments is reversed
pseudo_r2 = make_scorer(
    lambda y_true, y_pred: nmo.observation_models.PoissonObservations().pseudo_r2(
        y_pred, y_true
    )
)

# %% [markdown]
# #### Run the grid search:

# %%
gridsearch = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=pseudo_r2,
)

# run the 5-fold cross-validation grid search
gridsearch.fit(X, y)

# %% [markdown]
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

ax.set_xlabel("ridge regularization strength")
ax.set_ylabel("number of basis functions")

highlight_max_cell(cvdf_wide, ax)

# %% [markdown]
# #### Evaluating different values of the number of basis functions
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

# %% [markdown]
# ### Evaluating different bases directly

# %% [markdown]
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

# %% [markdown]
# Then run the grid search:

# %%
gridsearch = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=pseudo_r2,
)

# run the 5-fold cross-validation grid search
gridsearch.fit(X, y)

# %% [markdown]
# Wrangling the output data a bit and looking at the scores:

# %%
cvdf = pd.DataFrame(gridsearch.cv_results_)

# read out the number of
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

ax.set_xlabel("ridge regularization strength")
ax.set_ylabel("number of basis functions")

highlight_max_cell(cvdf_wide, ax)

# %% [markdown]
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

# %% [markdown]
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
