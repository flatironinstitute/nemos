.. _nemos_glm_hmm:

GLM-HMM
-------

Classes for creating Hidden Markov models (HMMs) with Generalized Linear Models (GLMs) observations.

.. currentmodule:: nemos.glm_hmm

.. autosummary::
    :toctree: generated/glm_hmm
    :nosignatures:

    GLMHMM

Initialization
^^^^^^^^^^^^^^

Functions and classes that compute initial parameters, to be passed to ``GLMHMM.fit`` as ``init_params``.

.. autosummary::
    :toctree: generated/glm_hmm
    :nosignatures:

    uniform_initial_proba_init
    random_initial_proba_init
    dirichlet_initial_proba_init
    sticky_transition_proba_init
    uniform_transition_proba_init
    random_transition_proba_init
    dirichlet_transition_proba_init
    random_glm_params_init
    constant_parameter_init
    KMeansInitializerGLM
