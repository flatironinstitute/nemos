"""Custom FeaturePytree class.

Implement the ``FeaturePytree`` class that stores inputs as a dictionary of arrays.
These objects can be used as inputs to NeMoS models. The fitted coefficients will be stored as
dictionaries with matching keys, facilitating interpretation of the model output.
"""

from collections import UserDict

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .tree_utils import pytree_map_and_reduce


@register_pytree_node_class
class FeaturePytree(UserDict):
    """Pytree to represent GLM features.

    This object is essentially a dictionary with strings as its keys and
    n-dimensional array-like objects as its values. FeaturePytree objects can
    only have a depth of 1, and we allow joint slicing.

    This is intended to be used with jax.tree_util.tree_map and similar functionality.
    """

    def __init__(self, **kwargs):
        self._num_time_points = None
        super().__init__(kwargs)

    @property
    def shape(self):
        """Return the sample points, i.e. the shared dimension."""
        # Every value is an array with the same number of time points, so
        # that's the only shared shape.
        return (self._num_time_points,)

    def __len__(self):
        """Return the number of sample points."""
        # Same logic as shape
        return self._num_time_points

    @property
    def ndim(self):
        """Force slicing only on the time axis."""
        # You can slice into a FeaturePytree along one dimension (time)
        return 1

    def __setitem__(self, key, value):
        """Check arrays and set the values."""
        # All keys are strings
        if not isinstance(key, str):
            raise ValueError("Keys must be strings!")
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
                raise ValueError(
                    f"All arrays must have same number of time points, {self._num_time_points}"
                )
        super().__setitem__(key, value)

    def __getitem__(self, key):
        """Getitem using data dictionary keys."""
        # We can index into either the features, which have strs as their
        # keys...
        if isinstance(key, str):
            return self.data[key]
        # Or the time dimension
        else:
            return jax.tree_util.tree_map(lambda x: x[key], self)

    def __repr__(self):
        """Represent a FeaturePytree."""
        # Show the shape and data type of each array
        repr = [f"{k}: shape {v.shape}, dtype {v.dtype}" for k, v in self.data.items()]
        return "\n".join(repr)

    def __eq__(self, other):
        """Equality over a tree of arrays."""
        # if structure is different, pytree_map_and_reduce will return a ValueError
        if jax.tree_util.tree_structure(self) != jax.tree_util.tree_structure(other):
            return False
        return pytree_map_and_reduce(
            lambda x, y: jnp.array_equal(x, y), all, self, other
        )

    def tree_flatten(self):
        """Flatten data."""
        return jax.tree_util.tree_flatten(self.data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the data dict and return the class."""
        try:
            return cls(**jax.tree_util.tree_unflatten(aux_data, children))
        except ValueError:
            return jax.tree_util.tree_unflatten(aux_data, children)
