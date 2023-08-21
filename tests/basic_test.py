from neurostatslib.glm import PoissonGLM
from sklearn.linear_model import PoissonRegressor
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from jax import grad
import scipy.stats as sts
import jax.numpy as jnp



np.random.seed(100)

nn, nt, ws, nb,nbi = 2, 15000, 30, 5, 0
X = np.random.normal(size=(nt, nn, nb*nn+nbi))
W_true = np.random.normal(size=(nn, nb*nn+nbi)) * 0.8
b_true = -3*np.ones(nn)
firing_rate = np.exp(np.einsum("ik,tik->ti", W_true, X) + b_true[None, :])
spikes = np.random.poisson(firing_rate)

# check likelihood
poiss_rand = sts.poisson(firing_rate)
mean_ll = poiss_rand.logpmf(spikes).mean()

# SKL FIT
weights_skl = np.zeros((nn, nb*nn+nbi))
b_skl = np.zeros(nn)
pred_skl = np.zeros((nt,nn))
for k in range(nn):
    model_skl = PoissonRegressor(alpha=0.,tol=10**-8,solver="lbfgs",max_iter=1000,fit_intercept=True)
    model_skl.fit(X[:,k,:], spikes[:, k])
    weights_skl[k] = model_skl.coef_
    b_skl[k] = model_skl.intercept_
    pred_skl[:, k] = model_skl.predict(X[:, k,:])


model_jax = PoissonGLM(score_type="pseudo-r2",solver_name="BFGS",
                solver_kwargs={'jit':True, 'tol': 10**-8, 'maxiter':1000},
                inverse_link_function=jnp.exp)
model_jax.fit(X, spikes)
mean_ll_jax = model_jax._score(X, spikes, (W_true, b_true))
firing_rate_jax = model_jax._predict((W_true, b_true),X)

print('jax pars - skl pars:', np.max(np.abs(model_jax.basis_coeff_ - weights_skl)))