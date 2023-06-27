import numpy as np
import pytest

@pytest.fixture
def initialize_basis():
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
def min_basis_funcs(basis_obj):
    min_basis = {
        'MSplineBasis': 1,
        'RaisedCosineBasisLinear': 1,
        'RaisedCosineBasisLog': 2,
        'OrthExponentialBasis': 1
    }
    if basis_obj.__class__.__name__ == 'CyclicBsplineBasis':
        min_basis['CyclicBsplineBasis'] = max(basis_obj._order * 2 - 2, basis_obj._order + 2)
    elif basis_obj.__class__.__name__ == 'CyclicBsplineBasis':
        min_basis['BSplineBasis'] = basis_obj._order + 2

    yield min_basis[basis_obj.__class__.__name__] <= basis_obj._n_basis_funcs
