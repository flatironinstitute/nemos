import nemos as nmo
import numpy as np


strength = dict(f1=np.array([1, 1, 1, 1, 1]), f2=np.array([0.5, 0.5]))
strength = dict(f1=1.0, f2=1.0)

X = nmo.pytrees.FeaturePytree(
    f1=np.random.normal(size=(100, 5)),
    f2=np.random.normal(size=(100, 2)),
)
y = np.random.randint(low=0, high=10, size=(100,))

glm = nmo.glm.GLM(
    regularizer=nmo.regularizer.Lasso(),
    regularizer_strength=strength,
    solver_name="ProxSVRG",
)
# glm = nmo.glm.GLM(
#    regularizer=nmo.regularizer.ElasticNet(), regularizer_strength=strength
# )
glm.fit(X, y)
