import numpy as np
import pytest
import neurostatslib.basis as basis

def pytest_generate_tests(metafunc):
    # called once per each test function
    if not (hasattr(metafunc.function, '__qualname__') and '.' in metafunc.function.__qualname__):
        # skip if not class
        return
    if not 'params' in metafunc.cls.__dict__:
        # skip if params is not defined
        return
    funcarglist = metafunc.cls.params[metafunc.function.__name__]

    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )

@pytest.fixture
def init_basis_parameter_grid():
    init_input_dict = {
        'MSplineBasis': [
            (nbasis, order) for nbasis in range(6,10) for order in range(1, 5)
        ],
        'RaisedCosineBasisLinear': [
            (nbasis, ) for nbasis in range(2,10)
        ],
        'RaisedCosineBasisLog': [
            (nbasis, ) for nbasis in range(2,10)
        ],
        'OrthExponentialBasis':[
            (nbasis, np.linspace(10, nbasis*10,nbasis)) for nbasis in range(6,10)
        ]
    }
    return init_input_dict



@pytest.fixture
def min_basis_funcs():
    min_basis = {}
    for spline in ['MSplineBasis']:
        for order in [-1, 0, 1, 2, 3, 4]:
            for n_basis in [-1, 0, 1, 3, 10, 20]:

                if spline == 'CyclicBSplineBasis':
                    raise_exception = (n_basis < max(order * 2 - 2, order + 2)) or (order < 2)
                elif spline == 'BSplineBasis':
                    raise_exception = n_basis < order + 2
                elif spline == 'MSplineBasis':
                    raise_exception = order < 1

                min_basis[spline] = {'args': {'order': order, 'n_basis_funcs': n_basis}, 'raise_exception': raise_exception}

    for n_basis in [-1, 0, 1, 3, 10, 20]:
        min_basis['RaisedCosineBasisLinear'] = {'args':{'n_basis_funcs': n_basis}, 'raise_exception': n_basis < 1}
        min_basis['RaisedCosineBasisLog'] = {'args':{'n_basis_funcs': n_basis}, 'raise_exception': n_basis < 2}
        min_basis['OrthExponentialBasis'] = {'args':{'n_basis_funcs': n_basis, 'decay_rates':np.linspace(0,1,max(1,n_basis))}, 'raise_exception': n_basis < 1}


    return min_basis


@pytest.fixture
def evaluate_basis_object():
    params = {
        'MSplineBasis': {'basis_obj': basis.MSplineBasis(10), 'n_input': 1},
        'RaisedCosineBasisLinear': {'basis_obj': basis.RaisedCosineBasisLinear(10), 'n_input': 1},
        'RaisedCosineBasisLog': {'basis_obj': basis.RaisedCosineBasisLog(10), 'n_input': 1},
        'OrthExponentialBasis': {'basis_obj': basis.OrthExponentialBasis(10, np.linspace(1, 10, 10)), 'n_input': 1},
        'add2': {'basis_obj': basis.MSplineBasis(10) + basis.MSplineBasis(10), 'n_input': 2},
        'mul2': {'basis_obj': basis.MSplineBasis(10) * basis.MSplineBasis(10), 'n_input': 2},
        'add3': {'basis_obj': basis.MSplineBasis(10) + basis.MSplineBasis(10) + basis.MSplineBasis(10), 'n_input': 3}

    }

    return params

@pytest.fixture
def basis_sample_consistency_check():
    params = {
        'add2':
            {'basis_obj': basis.MSplineBasis(10) + basis.MSplineBasis(10), 'n_input': 2},
        'mul2':
            {'basis_obj': basis.MSplineBasis(10) * basis.MSplineBasis(10), 'n_input': 2},
        'add3':
            {'basis_obj': basis.MSplineBasis(10) + basis.MSplineBasis(10) + basis.MSplineBasis(10), 'n_input': 3}
    }
    return params
