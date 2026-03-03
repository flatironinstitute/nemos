import numpy as np
import pytest

from nemos.label_encoder import LabelEncoder


@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_encode_decode_roundtrip(n_classes):
    """Test that encoding then decoding returns original labels."""

    # Test with string labels
    y = np.repeat(np.arange(n_classes), 2)
    encoder = LabelEncoder(n_classes=n_classes)
    label = np.array([chr(i) for i in range(ord("a"), ord("a") + n_classes)])
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


def test_n_classes_none():
    encoder = LabelEncoder(2)
    encoder.n_classes = None
