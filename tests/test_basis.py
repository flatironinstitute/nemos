import inspect
import pytest
import neurostatslib.basis as basis
import utils_testing



def test_basis_abstract_method_compliance():
    utils_testing.check_all_abstract_methods_compliance(basis)
