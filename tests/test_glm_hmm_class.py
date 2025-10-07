import pytest

import nemos as nmo


@pytest.mark.parametrize(
    "instantiate_hmm_glm",
    ["Poisson", "Bernoulli", "NegativeBinomial", "Gamma"],
    indirect=True,
)
def test_get_fit_attrs(instantiate_hmm_glm):
    X, y, model, params, rates, latents = instantiate_hmm_glm
    expected_state = {
        "coef_": None,
        "glm_params_": None,
        "initial_prob_": None,
        "intercept_": None,
        "transition_prob_": None,
    }
    assert model._get_fit_state() == expected_state
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)
    assert all(val is not None for val in model._get_fit_state().values())
    assert model._get_fit_state().keys() == expected_state.keys()
