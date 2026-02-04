import nemos as nmo
import numpy as np

import jax.numpy as jnp

strength = dict(f1=np.array([1, 1, 1, 1, 1]), f2=np.array([0.5, 0.5]))

X = nmo.pytrees.FeaturePytree(
    f1=np.random.normal(size=(100, 5)),
    f2=np.random.normal(size=(100, 2)),
)
X = np.random.normal(size=(100,5))
y = np.random.randint(low=0, high=10, size=(100,))

# glm = nmo.glm.PopulationGLM(
#    regularizer=nmo.regularizer.UnRegularized(),
#    regularizer_strength=1.0,
# )
strength = (dict(f1=1.0, f2=0.5), dict(f1=1.0, f2=0.5))

glm = nmo.glm.GLM(regularizer=nmo.regularizer.Ridge(), regularizer_strength=0.1)
glm.fit(X, y)
