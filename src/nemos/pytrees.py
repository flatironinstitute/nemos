#!/usr/bin/env python3
from collections import UserDict

import jax
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class FeaturePytree(UserDict):
    """Pytree to represent GLM features.

    This object is essentially a dictionary with strings as its keys and
    n-dimensional array-like objects as its values. FeaturePytree objects can
    only have a depth of 1, and we allow joint slicing.

    This is intended to be used with jax.tree_map and similar functionality.

    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    @property
    def shape(self):
        # the only shape we know is the number of features we have, as returned
        # by len
        return (self.__len__(), )

    @property
    def ndim(self):
        # You can only slice into FeaturePytree along the time dimension, so it
        # has dimensionality one.
        return 1

    def __setitem__(self, key, value):
        # All keys are strings
        if not isinstance(key, str):
            raise ValueError("keys must be strings!")
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
        repr = []
        for k, v in self.data.items():
            try:
                repr.append(f'{k}: shape {v.shape}, dtype {v.dtype}')
            except AttributeError:
                repr.append(f'{k}: {v}')
        return '\n'.join(repr)

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**jax.tree_util.tree_unflatten(aux_data, children))
