# content of ./test_parametrize.py
import pytest
import numpy as np
from neurostatslib.basis import BSplineBasis, Cyclic_BSplineBasis

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


class TestBspline:
    # a map specifying multiple argument sets for a test method
    params = {'test_bspline': [], 'test_cyclic_bspline':[]}

    # fill the parameter grid for B-spline
    for n_basis in range(1, 10):
        for order in range(1, n_basis):
            for window_size in [11, 100, 1000]:
                params['test_bspline'].append({'n_basis': n_basis, 'order': order, 'window_size': window_size, 'seed':100})

    # fill the param grid for cyclic B-spline (respecting the constraint on order and n_basis)
    for n_basis in range(4, 12):
        for order in range(2, 6):
            if (n_basis < 2 * (order - 1)) or (n_basis < order + 2):
                continue
            for window_size in [11, 100, 1000]:
                params['test_cyclic_bspline'].append({'n_basis': n_basis, 'order': order, 'window_size': window_size, 'seed':100})

    def test_bspline(self, n_basis, order, window_size, seed):

        rng = np.random.RandomState(seed)
        sample_points = rng.normal(size=window_size+100)

        # create basis
        basis = BSplineBasis(n_basis, window_size, order)
        basis.generate_knots(sample_points, 0.05, 0.95, False)

        # evaluate on samples
        eval_basis = basis.gen_basis_funcs(sample_points, outer_ok=True, der=0)

        # check dimension match expected
        assert (eval_basis.shape[0] == n_basis)
        assert (eval_basis.shape[1] == sample_points.shape[0])

        # check that the basis is set to 0 outside knots range
        sel = (sample_points < basis.knot_locs[0]) | (sample_points > basis.knot_locs[-1])

        # check sum to 1 constraint within the knots range
        assert (all(np.abs(eval_basis[:, ~sel].sum(axis=0) - 1) < 1e-8))

        # check equal 0 outside the range
        assert (all(np.abs(eval_basis[:, sel].sum(axis=0)) == 0))

    def test_cyclic_bspline(self, n_basis, order, window_size, seed):
        rng = np.random.RandomState(seed)
        sample_points = rng.normal(size=window_size + 100)

        # create basis
        basis = Cyclic_BSplineBasis(n_basis, window_size, order)
        basis.generate_knots(sample_points, 0, 1, True)

        # evaluate on samples
        eval_basis = basis.gen_basis_funcs(sample_points, der=0)

        # check dimension match expected
        assert (eval_basis.shape[0] == n_basis)
        assert (eval_basis.shape[1] == sample_points.shape[0])

        # check sum to 1 constraint (cyclic assume range)
        assert (all(np.abs(eval_basis.sum(axis=0) - 1) < 1e-8))