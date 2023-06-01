import pytest
from neurostatslib.basis import BSplineBasis, Cyclic_BSplineBasis
import numpy as np

np.random.seed(10**5)

par_grid = []
for n_basis in range(1, 10):
    for order in range(1, n_basis):
        for window_size in [11,100,1000]:
            par_grid.append({'n_basis':n_basis, 'order':order, 'window_size':window_size})
samp = np.random.normal(size=1000)
@pytest.mark.parametrize('sample_points', [samp])
@pytest.mark.parametrize('parameters',par_grid)

def test_Bspline_properties(sample_points,parameters):

    # create basis
    basis = BSplineBasis(parameters['n_basis'], parameters['window_size'], parameters['order'])
    basis.generate_knots(sample_points, 0.05, 0.95, False)

    # evaluate on samples
    eval_basis = basis.gen_basis_funcs(sample_points, outer_ok=True, der=0)

    # check dimension match expected
    assert (eval_basis.shape[0] == parameters['n_basis'])
    assert (eval_basis.shape[1] == sample_points.shape[0])

    # check that the basis is set to 0 outside knots range
    sel = (sample_points < basis.knot_locs[0]) | (sample_points > basis.knot_locs[-1])

    # check sum to 1 constraint within the knots range
    assert (all(np.abs(eval_basis[:, ~sel].sum(axis=0) - 1) < 1e-8 ))

    # check equal 0 outside the range
    assert (all( np.abs(eval_basis[:, sel].sum(axis=0)) == 0))


par_grid = []
for n_basis in range(4, 12):
    for order in range(2, 6):
        if (n_basis < 2*(order-1)) or (n_basis < order + 2):
            continue
        for window_size in [11,100,1000]:
            par_grid.append({'n_basis':n_basis, 'order':order, 'window_size':window_size})

@pytest.mark.parametrize('sample_points', [samp])
@pytest.mark.parametrize('parameters',par_grid)
def test_Cyclic_Bspline_properties(sample_points, parameters):

    # create basis
    basis = Cyclic_BSplineBasis(parameters['n_basis'], parameters['window_size'], parameters['order'])
    basis.generate_knots(sample_points, 0, 1, True)

    # evaluate on samples
    eval_basis = basis.gen_basis_funcs(sample_points, der=0)
    # check dimension match expected
    assert (eval_basis.shape[0] == parameters['n_basis'])
    assert (eval_basis.shape[1] == sample_points.shape[0])

    # check sum to 1 constraint (cyclic assume range)
    assert (all(np.abs(eval_basis.sum(axis=0) - 1) < 1e-8 ))

