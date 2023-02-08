import numpy as np
from sklearn.linear_model import PoissonRegressor
from scipy.ndimage import convolve1d
from basis import MSpline
import torch
import torch.nn.functional as F


@torch.no_grad()
def broadcast_convolve(v, B):
    """
    v : vector (num_timebins)
    B : matrix (num_basis_funcs x num_timebins)
    """
    vt = len(v)
    nb, bt = B.shape
    return F.conv1d(
        torch.from_numpy(np.tile(v[None, :], (nb, 1))),
        torch.from_numpy(B[:, None, :]),
        bias=None, stride=1,
        groups=nb, padding=0
    ).numpy()


class GLM:

    def __init__(self, spike_basis, covariate_basis=None):
        self.spike_basis = spike_basis
        self.covariate_basis = covariate_basis

        # (num_basis_funcs x window_size)
        self._B = self.spike_basis.transform()

    def fit(self, spike_data, covariates=None):
        """
        Parameters
        ----------
        spike_data : array (num_neurons x num_timebins)
            Spike counts arranged in a matrix.

        covariates : array (num_covariates x num_timebins)
            Other input variables (e.g. stimulus features)
        """

        if covariates is not None:
            raise NotImplementedError()

        assert spike_data.ndim == 2

        # Number of neurons and timebins
        nn, nt = spike_data.shape

        # (num_neurons x num_basis_funcs x num_timebins)
        X = np.stack(
            [broadcast_convolve(neuron, self._B.T) for neuron in spike_data],
            axis=0
        )
        oo = nt - X.shape[-1]
        print(X.shape)


        # (num_neurons x num_basis_funcs)
        self.coupling_coef_ = np.empty((nn, self.spike_basis.num_basis_funcs, nn))
        self.coupling_intercept_ = np.empty(nn)

        # Define sklearn model.
        prg = PoissonRegressor(max_iter=1000)

        # Fit parameters one neuron at a time.
        for i, y in enumerate(spike_data):
            prg.fit(X.reshape(-1, nt - oo).T, y=spike_data[i, oo:])
            self.coupling_coef_[i] = prg.coef_.reshape(
                self.spike_basis.num_basis_funcs, nn
            )
            self.coupling_intercept_[i] = prg.intercept_

    def predict(self, spike_data, covariates=None):
        """
        Parameters
        ----------
        spike_data : array (num_neurons x num_timebins)
            Spike counts arranged in a matrix.

        covariates : array (num_covariates x num_timebins)
            Other input variables (e.g. stimulus features)
        """

        # TODO - add assert statement that self.fit(...) was called

        # (num_neurons x num_basis_funcs x num_timebins)
        X = np.stack(
            [broadcast_convolve(neuron, self._B.T) for neuron in spike_data],
            axis=0
        )
        return np.einsum("jbt,nbj->nt", X, self.coupling_coef_) + self.coupling_intercept_[:, None]


# Short test
if __name__ == "__main__":

    nn, nt = 10, 1000
    rs = np.random.RandomState(123)
    spike_data = rs.poisson(np.ones((nn, nt))).astype("float64")

    model = GLM(
        spike_basis=MSpline(num_basis_funcs=6, window_size=100, order=3),
        covariate_basis=None
    )

    model.fit(spike_data)
    model.predict(spike_data)
