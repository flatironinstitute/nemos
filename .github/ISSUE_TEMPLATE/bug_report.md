---
name: Bug report & Installation Issues
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Please provide a short, reproducible example of the error, for example:

```python
import nemos as nmo
bspline = nmo.basis.BSplineEval(5)
# This raises an error
bspline.compute_features(np.random.randn(10), np.random.randn(10))
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environemnt (please complete the following information):**
 - OS: [e.g. macOS]
 - Python version [e.g. 3.12]
 - JAX version [e.g., 0.5]
 - NeMoS version [e.g. 0.1]

**Additional context**
Add any other context about the problem here.
