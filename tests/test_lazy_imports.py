"""Tests to enforce lazy loading of optional heavy dependencies.

These tests ensure that importing nemos does not eagerly load optional
dependencies like pynapple, dandi, h5py, and pynwb. This keeps the base
import time fast for users who don't need these features.

To fix a failing test:
1. Find where the module is being imported eagerly
2. Use one of these patterns to make it lazy:

   a) For type hints only - use TYPE_CHECKING:
      ```python
      from typing import TYPE_CHECKING
      if TYPE_CHECKING:
          import pynapple as nap
      ```

   b) For runtime usage - use lazy_loader:
      ```python
      import lazy_loader as lazy
      nap = lazy.load("pynapple")
      ```

   c) For module-level constants that use the lazy module:
      ```python
      _CACHED_VALUE = None
      def _get_value():
          global _CACHED_VALUE
          if _CACHED_VALUE is None:
              _CACHED_VALUE = nap.some_config.value
          return _CACHED_VALUE
      ```

Note on lazy_loader behavior:
- lazy.load("module") adds a lazy wrapper to sys.modules immediately
- The actual module code only runs when an attribute is accessed
- Therefore we check for heavy sub-dependencies (e.g., numba for pynapple)
  rather than the top-level module name
"""

import subprocess
import sys

import pytest


def _check_module_not_imported_after(
    import_statement: str, module_name: str
) -> tuple[bool, str]:
    """Check if a module is imported after executing an import statement.

    Runs in a subprocess to ensure clean Python environment.

    Returns
    -------
    tuple[bool, str]
        (is_lazy, error_message)
    """
    code = f"""
import sys
{import_statement}
imported = '{module_name}' in sys.modules
print('IMPORTED' if imported else 'NOT_IMPORTED')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()
    is_lazy = output == "NOT_IMPORTED"

    if not is_lazy:
        error_msg = (
            f"Module '{module_name}' was imported after `{import_statement}`.\n"
            f"This slows down import time.\n\n"
            f"stderr: {result.stderr}\n"
        )
    else:
        error_msg = ""

    return is_lazy, error_msg


# =============================================================================
# Test 1: Base nemos import should not load heavy dependencies
# =============================================================================

# Check for heavy sub-dependencies to detect actual loading (not lazy wrappers)
# - pynapple -> numba (pynapple's heavy compiled dependency)
# - sklearn -> sklearn.base (sklearn doesn't use lazy loading)
# - dandi, h5py, pynwb, fsspec are checked directly (not lazy-wrapped by nemos)
LAZY_MODULES_BASE = [
    ("pynapple", "numba"),  # Check numba to detect pynapple actually loading
    ("sklearn", "sklearn"),  # sklearn doesn't use lazy loading
    ("dandi", "dandi"),
    ("h5py", "h5py"),
    ("pynwb", "pynwb"),
    ("fsspec", "fsspec"),
]


@pytest.mark.parametrize("module_name,check_module", LAZY_MODULES_BASE)
def test_base_import_lazy(module_name: str, check_module: str):
    """Test that `import nemos` does not load heavy dependencies."""
    is_lazy, error_msg = _check_module_not_imported_after("import nemos", check_module)
    assert is_lazy, (
        f"{module_name} (checked via {check_module}) was loaded by `import nemos`.\n"
        f"{error_msg}"
    )


def test_base_import_is_fast():
    """Test that base nemos import completes in reasonable time.

    This is a sanity check that lazy loading is working. The threshold
    is generous to avoid flaky tests on slow CI machines.
    """
    code = """
import time
t0 = time.time()
import nemos
elapsed = time.time() - t0
print(f'{elapsed:.3f}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    elapsed = float(result.stdout.strip())

    # Base import should be under 500ms (generous threshold for CI)
    # With lazy loading, it's typically ~35ms
    assert elapsed < 0.5, (
        f"Base `import nemos` took {elapsed:.2f}s, which is too slow.\n"
        f"Expected < 0.5s with lazy loading.\n"
        f"Check that all submodules are lazy-loaded in nemos/__init__.py"
    )


# =============================================================================
# Test 2: type_casting.support_pynapple should not load pynapple without data
# =============================================================================


def test_support_pynapple_decorator_no_pynapple_data():
    """Test that using support_pynapple decorator without pynapple data doesn't fully load pynapple.

    We check for numba (pynapple's heavy dependency) rather than pynapple itself,
    because lazy.load() adds a lazy wrapper to sys.modules but doesn't execute
    pynapple's code until an attribute is accessed.
    """
    code = """
import sys
from nemos.type_casting import support_pynapple
import numpy as np

@support_pynapple()
def my_func(x):
    return x * 2

# Call with numpy array (no pynapple)
result = my_func(np.array([1, 2, 3]))

# Check for numba - pynapple's heavy dependency that loads when pynapple actually runs
imported = 'numba' in sys.modules
print('IMPORTED' if imported else 'NOT_IMPORTED')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()
    is_lazy = output == "NOT_IMPORTED"

    assert is_lazy, (
        f"pynapple was fully loaded (numba in sys.modules) when using support_pynapple with numpy data.\n"
        f"The decorator should only load pynapple when pynapple objects are passed.\n"
        f"stderr: {result.stderr}"
    )


# =============================================================================
# Test 3: Modules that don't need sklearn should not import it
# =============================================================================

# These modules should work without loading sklearn
SKLEARN_FREE_MODULES = [
    "nemos.utils",
    "nemos.simulation",
    "nemos.identifiability_constraints",
    "nemos.convolve",
    "nemos.exceptions",
    "nemos.pytrees",
    "nemos.tree_utils",
    "nemos.type_casting",
    "nemos.fetch",
    "nemos.solvers",
]


@pytest.mark.parametrize("module_path", SKLEARN_FREE_MODULES)
def test_sklearn_not_loaded(module_path: str):
    """Test that modules that don't need sklearn don't import it."""
    is_lazy, error_msg = _check_module_not_imported_after(
        f"import {module_path}", "sklearn"
    )
    assert is_lazy, (
        f"sklearn was imported by `import {module_path}`.\n"
        f"This module should not depend on sklearn.\n"
        f"{error_msg}"
    )


# =============================================================================
# Test 4: Modules that don't need jax should not import it
# =============================================================================

# These modules should work without loading jax
JAX_FREE_MODULES = [
    "nemos.fetch",
    "nemos.exceptions",
]


@pytest.mark.parametrize("module_path", JAX_FREE_MODULES)
def test_jax_not_loaded(module_path: str):
    """Test that modules that don't need jax don't import it."""
    is_lazy, error_msg = _check_module_not_imported_after(
        f"import {module_path}", "jax"
    )
    assert is_lazy, (
        f"jax was imported by `import {module_path}`.\n"
        f"This module should not depend on jax.\n"
        f"{error_msg}"
    )


# =============================================================================
# Test 5: Modules that don't need pynapple should not fully load it
# =============================================================================

# These modules should work without fully loading pynapple
# We check for numba (pynapple's heavy dependency) rather than pynapple itself
PYNAPPLE_FREE_MODULES = [
    "nemos.utils",
    "nemos.simulation",
    "nemos.identifiability_constraints",
    "nemos.convolve",
    "nemos.exceptions",
    "nemos.pytrees",
    "nemos.tree_utils",
    "nemos.fetch",
    "nemos.solvers",
    "nemos.regularizer",
    "nemos.glm",
]


@pytest.mark.parametrize("module_path", PYNAPPLE_FREE_MODULES)
def test_pynapple_not_loaded(module_path: str):
    """Test that modules that don't need pynapple don't fully load it.

    We check for numba (pynapple's heavy dependency) rather than pynapple itself,
    because lazy.load() adds a lazy wrapper to sys.modules but doesn't execute
    pynapple's code until an attribute is accessed.
    """
    is_lazy, error_msg = _check_module_not_imported_after(
        f"import {module_path}", "numba"
    )
    assert is_lazy, (
        f"pynapple was fully loaded (numba in sys.modules) by `import {module_path}`.\n"
        f"This module should not depend on pynapple.\n"
        f"{error_msg}"
    )
