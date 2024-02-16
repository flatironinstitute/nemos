import pytest
import pynapple as nap
import jax.numpy as jnp
import numpy as np
import nemos as nmo



@pytest.mark.parametrize(
    "inp, jax_func, expected_type",
    [
        ([jnp.arange(3)], lambda x: jnp.power(x, 2), jnp.ndarray),
        ([np.arange(3)], lambda x: jnp.power(x, 2), jnp.ndarray),
        ([nap.Tsd(t=np.arange(3), d=np.arange(3))], lambda x: jnp.power(x, 2), nap.Tsd),
        ([nap.TsdFrame(t=np.arange(3), d=np.expand_dims(np.arange(3), 1))],
         lambda x: jnp.power(x, 2), nap.TsdFrame),
        ([nap.TsdTensor(t=np.arange(3), d=np.expand_dims(np.arange(3), (1, 2, 3)))],
         lambda x: jnp.power(x, 2), nap.TsdTensor),
        ([nmo.glm.GLM()], lambda x: x, nmo.glm.GLM),
        ([nap.Tsd(t=np.arange(3), d=np.arange(3)), np.arange(3)], lambda x, y: (x+y, y), (nap.Tsd, nap.Tsd)),
        ([nap.Tsd(t=np.arange(3), d=np.arange(3)), np.expand_dims(np.arange(3),1)],
         lambda x, y: (np.power(x, 2), x+y), (nap.Tsd, nap.TsdFrame)),
        ([nap.Tsd(t=np.arange(3), d=np.arange(3)), np.expand_dims(np.arange(3),(1,2))],
         lambda x, y: (np.power(x, 2), x+y), (nap.Tsd, nap.TsdTensor))
    ]
)
def test_decorator_output_type(inp, jax_func, expected_type):
    """Test the decorator functionality on different input types."""
    @nmo.type_casting.cast_jax
    def func(*x):
        return jax_func(*x)

    out = func(*inp)
    assert nmo.utils.pytree_map_and_reduce(lambda x,y: isinstance(x, y), all, out, expected_type)


@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.49, 1])
    ]
)
def test_predict_nap_output(iset, poissonGLM_model_instantiation):
    X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
    n_samp = X.shape[0]
    time = jnp.linspace(0, 1, n_samp)
    tsd_X = nap.TsdTensor(t=time, d=X).restrict(iset)
    tsd_y = nap.TsdFrame(t=time, d=y).restrict(iset)
    model.fit(X, y)
    pred_rate = model.predict(tsd_X)
    assert isinstance(pred_rate, type(tsd_y))
    assert np.all(iset.values == pred_rate.time_support.values)



