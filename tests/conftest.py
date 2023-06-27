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

