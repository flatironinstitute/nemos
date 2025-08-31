---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Creating custom solvers for use with NeMoS

To support flexibility and long-term maintenance, NeMoS now has a backend-agnostic solver interface, allowing the use of solvers from different backend libraries with different interfaces.  
This also means that users can provide their own solvers, and as long as they adhere to the interface defined by `AbstractSolver`, they should be compatible with NeMoS and can be used for fitting models.

In the following we will walk through how one can create a NeMoS-compatible solver that uses `scipy.optimize.minimize` in the background.

+++

# Define the `scipy` adapter

In order to adhere to the `AbstractSolver` interface, we have to define the following methods in our custom solver class:
- `__init__`: all solver parameters and settings should go here. The other methods only take the solver state, current or initial solution (model parameters), and the input data for the objective function.
- `init_state`: Initialize the solver state.
- `update`: Take one step of the optimization algorithm.
- `run`: Run a full optimization.
- `get_accepted_arguments`: Set of argument names that can be passed to `__init__`.
- `get_optim_info`: Collect diagnostic information about the optimization run into an `OptimizationInfo` namedtuple.

```{code-cell} ipython3
import jax

jax.config.update("jax_enable_x64", True)
```

```{code-cell} ipython3
import scipy.optimize
import numpy as np
from typing import NamedTuple
from nemos.solvers._abstract_solver import OptimizationInfo
from nemos.utils import get_flattener_unflattener


class ScipySolverState(NamedTuple):
    """State of a scipy-based solver."""

    # for keeping track of the number of steps when using update
    iter_num: int
    # res is for storing the results of the optimization run.
    # NOTE: Storing as a dict because for some reason scipy.optimize.OptimizeResult
    # doesn't work with update from the second iteration.
    res: dict


# NOTE: Omitting most types here for simplicity.
# For type hints of the interface, please see the `AbstractSolver` class.


class ScipySolver:
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        method: str | None = "L-BFGS-B",
        max_steps: int = 100,
        tol: float = 1e-8,
    ) -> None:
        """
        Required arguments to accept are `unregularized_loss`, `regularizer`, `regularizer_strength`.
        All parameters required for changing how the solver is created and run (except the input data) should come after these.
        These are the arguments we expose to the user and we will also need to list these in `get_accepted_arguments`.
        Users can set these when creating the model with e.g. `GLM(..., solver_kwargs={"max_steps" : 10})`.
        In our case these are stored and later passed to `scipy.optimize.minimize`:
        - `method`: name of the optimization method to use. We use "L-BFGS-B" here.
        - `max_steps`: maximum number of steps to take.
        - `tol`: tolerance for the convergence criteria.

        An alternative would be accept `**solver_kwargs`, store it, and pass anything in it to scipy.
        """
        # `regularizer.penalized_loss` returns a function which is the unregularized loss + a penalty term
        # we will pass this objective function to `scipy.optimize.minimize`
        self.fun = regularizer.penalized_loss(unregularized_loss, regularizer_strength)

        # storing these to later pass to scipy
        self.method = method
        self.tol = tol
        self.max_steps = max_steps

    @classmethod
    def get_accepted_arguments(cls):
        """
        Explicitly list the arguments we accept.
        For alternative, more flexible implementations see the solvers
        implemented in `nemos.solvers`.
        """
        return {"method", "max_steps", "tol"}

    def init_state(self, init_params, *args):
        """
        We create a ScipySolverState where:
          - `iter_num` is 0 as we haven't taken any steps yet.
          - `res` an empty dictionary as we don't have any results yet.

        Has to accept `init_params` and `*args` even if it doesn't use them.
        """
        return ScipySolverState(0, {})

    def run(self, init_params, *args):
        """
        Run a whole optimization by calling the helper function `_run_for_n_steps`
        defined below with the maximum number of steps allowed.

        Returns a tuple of the final parameters and the final solver state.
        """
        unstacked_final_params, res = self._run_for_n_steps(
            init_params, self.max_steps, *args
        )
        # unpack res, which has a custom type defined in scipy into a regular dict
        return unstacked_final_params, ScipySolverState(res.nit, {**res})

    def update(self, params, state, *args):
        """
        While scipy doesn't expose individual steps, we hack our way around that
        by passing `options = {"maxiter" : 1}` to `scipy.optimize.minimize`
        and increment the number steps in the state by hand.
        """
        unstacked_final_params, res = self._run_for_n_steps(params, 1, *args)
        # 0 or 1. Hopefully 1.
        assert res.nit <= 1
        return unstacked_final_params, ScipySolverState(
            state.iter_num + res.nit, {**res}
        )

    def _run_for_n_steps(self, params, n_steps: int, *args):
        """
        Helper function so that we're not duplicating almost the same code in `run` and `update`.

        Prepare things in the form `scipy.optimize.minimize` expects them, then call it to
        run the optimization for `n_steps` steps.
        """
        # `params` is a tuple of (weight_matrix, intercept), but `scipy.optimize.minimize` only
        # accepts a 1D vector of parameters, so flatten all parameters and concatenate them
        flattener_fun, unflattener_fun = get_flattener_unflattener(params)
        _flat_params = flattener_fun(params)

        # create an objective function that translates from the flat parameters accepted by scipy
        # back to their original form accepted by nemos.glm.GLM's objective function
        # so that it can be evaluated
        def _flat_obj(flat_params, *args):
            params_in_their_orig_shape = unflattener_fun(flat_params)
            return self.fun(params_in_their_orig_shape, *args)

        # pass the flat parameters and the objective function taking them
        # also the custom parameters we saved in __init__
        res = scipy.optimize.minimize(
            fun=_flat_obj,
            x0=_flat_params,
            args=args,
            method=self.method,
            options={"maxiter": n_steps},
            tol=self.tol,
        )

        return unflattener_fun(res.x), res

    def get_optim_info(self, state):
        """
        Read out the relevant optimization info from the results
        we stored in the state.
        """
        return OptimizationInfo(
            state.res["fun"],
            state.res["nit"],
            state.res["success"],
            state.res["nit"] >= self.max_steps,
        )
```

# Using `ScipySolver` for model fitting

+++

## Overwriting the registry

Currently, the solver registry defines which implementation to use for each algorithm, so that has to be overwritten in order to tell NeMoS to use a custom class.

This is hacky and not an intended use-case for now, but in the future we are [planning to support passing any solver to `BaseRegressor`](https://github.com/flatironinstitute/nemos/issues/378).

So we modify the solver registry to use this implementation when asking for L-BFGS:

```{code-cell} ipython3
from nemos.solvers._solver_registry import solver_registry

solver_registry["LBFGS"] = ScipySolver
```

## Generate toy data

```{code-cell} ipython3
import nemos as nmo
import numpy as np
from sklearn.datasets import make_regression

# num_samples, num_features, num_outputs = 100, 10, 3
num_samples, num_features, num_outputs = 100, 10, 1

# Generate a design matrix
X, y = make_regression(
    n_features=num_features,
    n_samples=num_samples,
    noise=10.0,
    n_targets=num_outputs,
)
# make spike counts
y = np.exp(y / y.max(axis=0) * 2 + 1)
y = np.random.poisson(y)
y = y.astype(np.float64)

# for multiple neurons, use PopulationGLM
if num_outputs > 1:
    glm_class = nmo.glm.PopulationGLM
else:
    glm_class = nmo.glm.GLM
```

## Create the model and fit 

As the solver registry returns our custom `ScipySolver` class when specifying `solver_name="LBFGS"`, `model.fit` will call `ScipySolver.run`. 

```{code-cell} ipython3
model = glm_class(
    solver_name="LBFGS",
    solver_kwargs={
        "max_steps": 100,
    },
)

model.fit(X, y)
```

We can inspect the model to show that it is using `ScipySolver`:

```{code-cell} ipython3
print(model._solver)
print(model._solver_run)
```

## Test the update method

`model.fit` called `ScipySolver.run` to perform a whole optimization and return the final solution.

Repeatedly calling a `model.update` calls `ScipySolver.update` to perform a single step of the optimization. While this is usually much slower, this way we can follow the evolution of the loss function's value or the model parameters.

This also showcases how quickly the L-BFGS algorithm used by `scipy` converges on this problem.

```{code-cell} ipython3
import matplotlib.pyplot as plt

params = model.initialize_params(X, y)
state = model.initialize_state(X, y, params)

fig, ax = plt.subplots()
scores = []
for i in range(20):
    params, state = model.update(params, state, X, y)
    scores.append(model.score(X, y))
ax.plot(range(1, 21), scores)
ax.set(xlabel="Iteration", ylabel="Obj. fun. value")
```

