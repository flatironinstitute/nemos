from typing import NamedTuple, Optional

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
        hess=None,
        maxiter: int = 100,
        tol: float = 1e-8,
        options: Optional[dict] = None,
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
        import jax

        # `regularizer.penalized_loss` returns a function which is the unregularized loss + a penalty term
        # we will pass this objective function to `scipy.optimize.minimize`
        self.fun = regularizer.penalized_loss(
            unregularized_loss, init_params, regularizer_strength
        )
        self.val_and_grad = jax.value_and_grad(self.fun)
        self.hess = hess
        # storing these to later pass to scipy
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.options = options

        if has_aux:
            raise NotImplementedError(
                "Auxiliary outputs are not supported by ScipySolver."
            )

    @classmethod
    def get_accepted_arguments(cls):
        """
        Explicitly list the arguments we accept.
        For alternative, more flexible implementations see the solvers
        implemented in `nemos.solvers`.
        """
        return {"method", "hess", "maxiter", "tol", "options"}

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
            init_params, self.maxiter, *args
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
        if res.nit > 1:
            raise RuntimeError(f"Expected at most 1 solver step, got {res.nit}.")
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
            f, fgrad = self.val_and_grad(params_in_their_orig_shape, *args)
            return f, flattener_fun(fgrad)

        def _flat_hess(flat_params, *args):
            params_in_their_orig_shape = unflattener_fun(flat_params)
            return self.hess(params_in_their_orig_shape, *args)

        # pass the flat parameters and the objective function taking them
        # also the custom parameters we saved in __init__
        options = dict(self.options or {})
        options.update({"maxiter": n_steps})
        res = scipy.optimize.minimize(
            fun=_flat_obj,
            jac=True,
            hess=_flat_hess if self.hess is not None else None,
            x0=_flat_params,
            args=args,
            method=self.method,
            options=options,
            tol=self.tol,
        )

        return unflattener_fun(res.x), res


class ScipyLBFGS(ScipySolver):
    """Solver using the L-BFGS-B algorithm via scipy.optimize.minimize.

    Stopped on the projected gradient alone. `scipy.optimize.minimize` routes its
    `tol` onto both `ftol` and `gtol` for L-BFGS-B and stops on whichever fires
    first. `ftol` tests the *relative decrease per iteration*, so it halts on
    stalled progress rather than on stationarity: on the head-direction benchmark
    it fires while the gradient sup-norm is still 2e-2. Pinning `ftol=0` leaves
    `gtol` as the only stopping rule, which is the same criterion the other
    solvers in the grid use, so timings and iteration counts are comparable.
    """

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params,
        maxiter: int = 100,
        tol: float = 1e-8,
        options: Optional[dict] = None,
    ):
        return super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            has_aux,
            init_params,
            method="L-BFGS-B",
            maxiter=maxiter,
            tol=tol,
            options={**(options or {}), "ftol": 0.0},
        )
