import nemos as nmo
import numpy as np

import jax.numpy as jnp

strength = dict(f1=np.array([1, 1, 1, 1, 1]), f2=np.array([0.5, 0.5]))
strength = dict(f1=1.0, f2=1.0)

X = nmo.pytrees.FeaturePytree(
    f1=np.random.normal(size=(100, 5)),
    f2=np.random.normal(size=(100, 2)),
)
y = np.random.randint(low=0, high=10, size=(100, 2))

glm = nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.UnRegularized(),
    regularizer_strength=1.0,
)
# glm = nmo.glm.GLM(
#    regularizer=nmo.regularizer.GroupLasso(mask=np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]])),
#    regularizer_strength=[np.array([1.0, 1.0])],
# )
glm.fit(np.random.normal(size=(100, 5)), y)
