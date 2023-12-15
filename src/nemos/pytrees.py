#!/usr/bin/env python3
import jax
from collections import UserDict


class FeaturePytree(UserDict):
    """Pytree to represent GLM features.

    This object is essentially a dictionary with strings as its keys and
    n-dimensional array-like objects as its values. We ensure that the arrays
    all have the same number of time points and allow joint slicing.

    This is intended to be used with jax.tree_map and similar functionality.

    Once you have finished using your FeaturePytree instance with nemos
    objects, you should grab the data attribute in order to get a regular
    dictionary representation.

    WARNING: The values of FeaturePytree must be array-like objects with at
    least 1 dimension (i.e., `v.shape[0]` is an int). In particular, this means
    you cannot tree_map a function which converts the type of the values or
    returns a scalar. For example, the following fails:

    >>> import nemos
    >>> d = nemos.pytrees.FeaturePytree(a=np.random.rand(10), b=np.random.rand(10))
    >>> jax.tree_map(lambda x: x.shape, d)

    As will this:

    >>> import nemos
    >>> d = nemos.pytrees.FeaturePytree(a=np.random.rand(10), b=np.random.rand(10))
    >>> jax.tree_map(lambda x: x.sum(), d)

    BUT the following will work:

    >>> import nemos
    >>> d = nemos.pytrees.FeaturePytree(a=np.random.rand(10, 1), b=np.random.rand(10, 1))
    >>> jax.tree_map(lambda x: jnp.dot(x.T, x), d)

    Because jnp.dot(x.T, x) returns an array of shape (1, 1).

    """
    def __init__(self, **kwargs):
        self._num_time_points = None
        super().__init__(kwargs)

    @property
    def shape(self):
        # Every value is an array with the same number of time points, so
        # that's the only shared shape.
        return (self._num_time_points, )

    def __len__(self):
        # Same logic as shape
        return self._num_time_points

    @property
    def ndim(self):
        # You can only slice into FeaturePytree along the time dimension, so it
        # has dimensionality one.
        return 1

    def __setitem__(self, key, value):
        # All keys are strings
        if not isinstance(key, str):
            raise ValueError("keys must be strings!")
        # All values are array-like and must have time points along the first
        # dimension, so double-check that's the case here.
        try:
            value.shape[0]
        except (AttributeError, IndexError):
            raise ValueError("All values must be arrays of at least 1 dimension!")
        # For the first value, we must set how time points this FeaturePytree
        # has
        if self._num_time_points is None:
            self._num_time_points = value.shape[0]
        # For all subsequent values, ensure they have the proper number of time points
        else:
            if self._num_time_points != value.shape[0]:
                raise ValueError(f"All arrays must have same number of time points, {self._num_time_points}")
        super().__setitem__(key, value)

    def __getitem__(self, key):
        # We can index into either the features, which have strs as their
        # keys...
        if isinstance(key, str):
            return self.data[key]
        # Or the time dimension
        else:
            return jax.tree_map(lambda x: x[key], self.data)

    def __repr__(self):
        # Show the shape and data type of each array
        repr = [f'{k}: shape {v.shape}, dtype {v.dtype}' for k, v in self.data.items()]
        return '\n'.join(repr)

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
         try:
            return cls(**jax.tree_util.tree_unflatten(aux_data, children))
         except ValueError:
            raise ValueError("It looks like you are using jax.tree_map with a function that doesn't return an array of at least 1 dimension (e.g., jnp.sum). This is unsupported. If you need to tree_map this function, consider applying it on the data attribute")
