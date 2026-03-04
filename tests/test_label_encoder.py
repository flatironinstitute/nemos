from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import numpy as np
import pytest

from nemos.label_encoder import LabelEncoder


@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_encode_decode_roundtrip(n_classes):
    """Test that encoding then decoding returns original labels."""

    # Test with string labels
    y = np.repeat(np.arange(n_classes), 2)
    encoder = LabelEncoder(n_classes=n_classes)
    label = np.array(["a", "b", "c"])[:n_classes]
    encoder.set_classes(label)
    y_label = encoder.decode(y)

    # Roundtrip: decode -> encode should give original indices
    y_roundtrip = encoder.encode(y_label)
    assert np.array_equal(y, y_roundtrip)

    # Roundtrip: encode -> decode should give original labels
    y_label_roundtrip = encoder.decode(encoder.encode(y_label))
    assert np.array_equal(y_label, y_label_roundtrip)


def test_classes_none_initially():
    """Test that classes_ is None before set_classes is called."""
    encoder = LabelEncoder(2)
    assert encoder.classes_ is None
    assert encoder._skip_encoding is False
    assert encoder._class_to_index_ is None


def test_reset_at_n_classes_set():
    encoder = LabelEncoder(2)
    encoder.set_classes([2, 1])
    assert encoder.classes_ is not None
    assert encoder._class_to_index_ is not None
    assert encoder._skip_encoding is False
    # see if reset is triggered
    encoder.n_classes = 4
    assert encoder.classes_ is None
    assert encoder._skip_encoding is False
    assert encoder._class_to_index_ is None


def test_set_classes_behavior():
    encoder = LabelEncoder(4)
    encoder.set_classes([0, 1, 2, 3])
    assert encoder.classes_ is not None
    assert encoder._class_to_index_ is None
    assert encoder._skip_encoding is True

    # check that we need an encoding
    # for custom labels
    encoder.set_classes([1, 2, 3, 4])
    assert encoder.classes_ is not None
    assert encoder._class_to_index_ is not None
    assert encoder._skip_encoding is False

    # check that raises
    with pytest.raises(ValueError, match=r"Found \d unique"):
        encoder.set_classes([0, 1, 2, 3, 4])

    with pytest.raises(ValueError, match="Found only"):
        encoder.set_classes([0, 1, 2])


@pytest.mark.parametrize(
    "labels, skip_encoding",
    [
        (jnp.array([1, 0, 2]), True),
        (jnp.array([2, 3, 4]), False),
    ],
)
def test_set_classes_jax(labels, skip_encoding):
    """Test set_classes with JAX arrays: default labels skip encoding, non-default don't."""
    encoder = LabelEncoder(3)
    encoder.set_classes(labels)
    assert encoder._skip_encoding is skip_encoding
    assert encoder.classes_ is not None


def test_encode_decode_roundtrip_jax():
    """Roundtrip with JAX integer arrays and non-default labels."""
    encoder = LabelEncoder(3)
    encoder.set_classes(jnp.array([2, 3, 4]))
    y = jnp.array([2, 3, 4, 2])
    assert jnp.array_equal(encoder.decode(encoder.encode(y)), y)


@pytest.mark.parametrize(
    "y, classes, safe",
    [
        (np.array([2, 3, 99]), np.array([2, 3, 4]), True),
        (jnp.array([2, 3, 99]), jnp.array([2, 3, 4]), True),
        (np.array([2, 3, 99]), np.array([2, 3, 4]), False),
        (jnp.array([2, 3, 99]), jnp.array([2, 3, 4]), False),
        (
            np.array(["a", "b", "z"]),
            np.array(["a", "b", "c"]),
            True,
        ),
        (np.array(["a", "b", "z"]), np.array(["a", "b", "c"]), False),
    ],
)
def test_encode_safe_flag_invalid_label(y, classes, safe):
    """safe=True raises on unrecognized labels; safe=False silently accepts them."""
    encoder = LabelEncoder(3)
    encoder.set_classes(classes)
    if safe:
        expectation = pytest.raises((KeyError, ValueError))
    else:
        expectation = does_not_raise()
    with expectation:
        encoder.encode(y, safe=safe)


@pytest.mark.parametrize(
    "y",
    [
        np.array([2, 3, 4, 2]),
        jnp.array([2, 3, 4, 2]),
        np.array(["a", "b", "c", "a"]),
    ],
)
def test_encode_safe_unsafe_agree_on_valid(y):
    """safe=True and safe=False produce identical results for valid labels."""
    encoder = LabelEncoder(3)
    encoder.set_classes(y[:-1])
    assert np.array_equal(encoder.encode(y, safe=True), encoder.encode(y, safe=False))


@pytest.mark.parametrize(
    "y, classes_unsorted, classes_sorted",
    [
        (np.array([2, 3, 4, 2]), np.array([4, 2, 3]), np.array([2, 3, 4])),
        (jnp.array([2, 3, 4, 2]), jnp.array([4, 2, 3]), jnp.array([2, 3, 4])),
    ],
)
def test_set_classes_sorts(y, classes_unsorted, classes_sorted):
    """set_classes sorts labels via unique: unsorted input gives same classes_ and encoding as sorted."""
    encoder_unsorted = LabelEncoder(3)
    encoder_unsorted.set_classes(classes_unsorted)
    encoder_sorted = LabelEncoder(3)
    encoder_sorted.set_classes(classes_sorted)
    assert np.array_equal(encoder_unsorted.classes_, encoder_sorted.classes_)
    assert np.array_equal(encoder_unsorted.encode(y), encoder_sorted.encode(y))


@pytest.mark.parametrize(
    "y, safe, match",
    [
        # out-of-range integer: safe=True raises, safe=False does not
        (np.array([0, 1, 5]), True, "Unrecognized label"),
        (jnp.array([0, 1, 5]), True, "Unrecognized label"),
        (np.array([0, 1, 5]), False, None),
        (jnp.array([0, 1, 5]), False, None),
        # non-integer dtype: safe=True raises with dtype message
        (np.array([0.0, 1.0, 2.0]), True, "Expected integer"),
        (np.array(["a", "b", "c"]), True, "Expected integer"),
        # valid default labels: safe=True does not raise
        (np.array([0, 1, 2]), True, None),
        (jnp.array([0, 1, 2]), True, None),
    ],
)
def test_encode_safe_canonical(y, safe, match):
    """safe=True validates range and dtype when classes are the default [0, n_classes-1]."""
    encoder = LabelEncoder(3)
    encoder.set_classes(np.array([0, 1, 2]))
    if safe and match:
        with pytest.raises(ValueError, match=match):
            encoder.encode(y, safe=safe)
    else:
        encoder.encode(y, safe=safe)


def test_check_classes_behavior():
    encoder = LabelEncoder(4)
    with pytest.raises(RuntimeError, match=r"Classes are not set.+hello.+"):
        encoder.check_classes_is_set("hello")
    # set and retry
    encoder.set_classes(["0", "1", "2", "3"])
    encoder.check_classes_is_set("hello")
