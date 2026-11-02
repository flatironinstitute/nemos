"""Read-only dictionary that supports pickling."""

from collections.abc import Mapping


class ReadOnlyDict(Mapping):
    """Simple dictionary-like immutable that supports read-only access.

    This is a simple read-only dictionary that can be used as
    an immutable class-attribute. Similar to `ReadOnlyDict(d)`,
    but supporting pickling.
    """

    def __init__(self, d: "dict | ReadOnlyDict"):
        # this raises when trying to store an array
        # which is what we want since arrays can be mutated
        self._hash = hash(frozenset(d.items()))
        self.d = d

    def __getitem__(self, key):
        return self.d[key]

    def __iter__(self):
        return iter(self.d)

    def __len__(self):
        return len(self.d)

    def __repr__(self):
        dict_repr = repr(self.d)
        return f"{self.__class__.__name__}({dict_repr})"

    def __hash__(self):
        """Needed for immutability."""
        return self._hash
