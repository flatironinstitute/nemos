VALID_BASE_REGRESSOR_METHODS = ["set_params", "get_params"]


def skip_external_methods(app, what, name, obj, skip, options):
    """Skip sklearn-inherited methods for classes in configured modules.

    Controlled via ``nemos_skip_sklearn_for_modules`` in ``conf.py``.
    """
    if what != "method":
        return None  # deferr skip behavior

    obj_module = getattr(obj, "__module__", "") or ""
    if not obj_module.startswith("sklearn"):
        return None

    skip = name not in VALID_BASE_REGRESSOR_METHODS
    if skip:
        print(f"skipping {name} for module {obj_module}")

    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_external_methods)
