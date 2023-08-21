import pytest

import jaxopt

import neurostatslib as nsl

class TestPoissonGLM:

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "ScipyMinimize", "NotPresent"])
    def test_initialization_solver_name(self, solver_name: str):
        try:
            getattr(jaxopt, solver_name)
            raise_exception = False
        except:
            raise_exception = True
        if raise_exception:
            with pytest.raises(AttributeError, match="module jaxopt has no attribute"):
                nsl.glm.PoissonGLM(solver_name=solver_name)
        else:
            nsl.glm.PoissonGLM(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "ScipyMinimize"])
    @pytest.mark.parametrize("solver_kwargs", [
                             {"tol":1, "verbose":1, "maxiter":1},
                             {"tol":1, "maxiter":1}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        raise_exception = (solver_name == "ScipyMinimize") & ("verbose" in solver_kwargs)
        if raise_exception:
            with pytest.raises(NameError, match="kwargs {'[a-z]+'} in solver_kwargs not a kwarg"):
                nsl.glm.PoissonGLM(solver_name, solver_kwargs=solver_kwargs)
        else:
            # define glm and instantiate the solver
            nsl.glm.PoissonGLM(solver_name, solver_kwargs=solver_kwargs)
            getattr(jaxopt, solver_name)(fun=lambda x: x, **solver_kwargs)
    def test_fit(self):
        pass


    def test_score(self):
        pass


    def test_predict(self):
        pass


    def test_simulate(self):
        pass

    def test_compare_to_scikitlearn(self):
        pass


