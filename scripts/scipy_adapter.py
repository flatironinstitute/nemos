from typing import NamedTuple

import scipy.optimize

from nemos.utils import get_flattener_unflattener


class ScipySolverState(NamedTuple):
    """State of a scipy-based solver."""

    # for keeping track of the number of steps when using update
    iter_num: int
    # res is for storing the results of the optimization run.
    res: dict
    # for notifying nemos if the optimization was succesful
    converged: bool


class ScipySolver:
    """General adapter class for using scipy.minimize as a NeMoS solver."""

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params,
        method: str,
        max_steps: int = 100,
        tol: float = 1e-8,
    ) -> None:
        """
        Required arguments to accept are `unregularized_loss`, `regularizer`, `regularizer_strength`.
        All parameters required for changing how the solver is created and run (except the input data) should come after these.
        These are the arguments we expose to the user and we will also need to list these in `get_accepted_arguments`.
        Users can set these when creating the model with e.g. `GLM(..., solver_kwargs={"max_steps" : 10})`.
        In our case these are stored and later passed to `scipy.optimize.minimize`:
        - `method`: name of the optimization method to use.
        - `max_steps`: maximum number of steps to take.
        - `tol`: tolerance for the convergence criteria.

        An alternative would be accept `**solver_kwargs`, store it, and pass anything in it to scipy.
        """
        # `regularizer.penalized_loss` returns a function which is the unregularized loss + a penalty term
        # we will pass this objective function to `scipy.optimize.minimize`
        self.fun = regularizer.penalized_loss(
            unregularized_loss, init_params, regularizer_strength
        )

        # storing these to later pass to scipy
        self.method = method
        self.tol = tol
        self.max_steps = max_steps

        assert not has_aux, "Not implementing aux support now."

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
        return ScipySolverState(0, {}, False)

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
        return (
            unstacked_final_params,
            ScipySolverState(res.nit, {**res}, res.success),
            None,
        )

    def update(self, params, state, *args):
        """
        While scipy doesn't expose individual steps, we hack our way around that
        by passing `options = {"maxiter" : 1}` to `scipy.optimize.minimize`
        and increment the number steps in the state by hand.
        """
        unstacked_final_params, res = self._run_for_n_steps(params, 1, *args)
        assert res.nit <= 1
        return (
            unstacked_final_params,
            ScipySolverState(state.iter_num + res.nit, {**res}, False),
            None,
        )

    def _run_for_n_steps(self, params, n_steps: int, *args):
        """
        Helper function so that we're not duplicating almost the same code in `run` and `update`.

        Prepare things in the form `scipy.optimize.minimize` expects them, then call it to
        run the optimization for `n_steps` steps.
        """
        # `params` returned by GLM is a tuple of (weight_matrix, intercept),
        # but `scipy.optimize.minimize` only accepts a 1D vector of parameters,
        # so flatten all parameters and concatenate them

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


class ScipyLBFGS(ScipySolver):
    """Solver using the modified Powell algorithm."""

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params,
        max_steps: int = 100,
        tol: float = 1e-8,
    ):
        return super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            has_aux,
            init_params,
            "L-BFGS-B",
            max_steps,
            tol,
        )
