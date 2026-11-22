import nemos as nmo
from nemos.base_regressor import BaseRegressor


def test_gpglm_importable():
    assert hasattr(nmo.gpglm, "GPGLM")


def test_gpglm_is_base_regressor():
    assert issubclass(nmo.gpglm.GPGLM, BaseRegressor)


def test_gpglm_instantiates():
    model = nmo.gpglm.GPGLM()
    assert isinstance(model, nmo.gpglm.GPGLM)
