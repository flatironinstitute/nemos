import pytest
import jax
import jax.numpy as jnp
import numpy as np
from nemos.pytrees import FeaturePytree
import nemos as nmo



class TestFeaturePytree:

    def test_key_error_init(self):
        # this actually isn't our error, but a standard python one.
        with pytest.raises(TypeError, match="keywords must be strings"):
            FeaturePytree(**{1: np.random.rand(100)})

    def test_key_error_set(self):
        tree = FeaturePytree(**{'test': np.random.rand(100)})
        with pytest.raises(ValueError, match="Keys must be strings"):
            tree[1] = np.random.rand(100)

    def test_num_timepoints_error_init(self):
        with pytest.raises(ValueError, match="All arrays must have same number of time points"):
            FeaturePytree(test1=np.random.rand(100, 1),
                          test2=np.random.rand(50, 1))

    def test_num_timepoints_error_set(self):
        tree = FeaturePytree(test1=np.random.rand(100, 1))
        with pytest.raises(ValueError, match="All arrays must have same number of time points"):
            tree['test2'] = np.random.rand(50, 1)

    def test_array_error_init(self):
        with pytest.raises(ValueError, match="All values must be arrays"):
            FeaturePytree(test1='hi')

    def test_array_error_set(self):
        tree = FeaturePytree(test1=np.random.rand(100, 1))
        with pytest.raises(ValueError, match="All values must be arrays"):
            tree['test2'] = 'hi'

    def test_diff_shapes(self):
        tree = FeaturePytree(test=np.random.rand(100))
        for dim in [1, 2, 3, 4]:
            tree[f'test{dim}'] = np.random.rand(100, dim)
        assert len(tree) == 100

    def test_diff_dims(self):
        tree = FeaturePytree(test=np.random.rand(100))
        for ndim in [1, 2, 3, 4]:
            tree[f'test{ndim}'] = np.random.rand(100, *[1]*ndim)
        assert len(tree) == 100

    def test_treemap(self):
        tree = FeaturePytree(test=np.random.rand(100, 1),
                             test2=np.random.rand(100, 2))
        mapped = jax.tree_map(lambda x: jnp.mean(x, axis=-1), tree)
        assert len(tree) == len(mapped)
        assert list(tree.keys()) == list(mapped.keys())

    def test_treemap_npts(self):
        tree = FeaturePytree(test=np.random.rand(100, 1),
                             test2=np.random.rand(100, 2))
        mapped = jax.tree_map(lambda x: x[::10], tree)
        assert len(mapped) == 10
        assert list(tree.keys()) == list(mapped.keys())

    def test_treemap_to_dict(self):
        tree = FeaturePytree(test=np.random.rand(100,),
                             test2=np.random.rand(100, 2))
        mapped = jax.tree_map(jnp.mean, tree)
        assert isinstance(mapped, dict)
        assert list(tree.keys()) == list(mapped.keys())

    def test_get_key(self):
        test = np.random.rand(100, 1)
        tree = FeaturePytree(test=test)
        np.testing.assert_equal(tree['test'], test)
        with pytest.raises(KeyError):
            tree['hi']

    def test_get_slice(self):
        tree = FeaturePytree(test=np.random.rand(100, 1),
                             test2=np.random.rand(100, 2))
        assert len(tree[:10]) == 10
        assert list(tree.keys()) == list(tree[:10].keys())

    def test_glm(self):
        w_true = FeaturePytree(test=np.random.rand(1, 3),
                               test2=np.random.rand(1, 2))
        X = FeaturePytree(test=np.random.rand(100, 1, 3),
                          test2=np.random.rand(100, 1, 2))
        rate = nmo.utils.pytree_map_and_reduce(lambda w, x: jnp.einsum("ik,tik->ti", w, x),
                                               sum, w_true, X)
        spikes = np.random.poisson(rate)
        model = nmo.glm.GLM()
        model.fit(X, spikes)
        assert list(model.coef_.keys()) == list(X.keys())
        for k in model.coef_.keys():
            assert model.coef_[k].shape == X[k].shape[1:]
