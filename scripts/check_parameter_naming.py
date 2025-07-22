import difflib
import inspect
import itertools
import logging
import sys
import types
from typing import Optional

# Pairs of parameter names that are lexically similar but intentionally allowed.

# During parameter name similarity checks, some pairs of names may be flagged
# as potentially inconsistent due to their high string similarity. This list
# enumerates such known, acceptable pairs that should be *excluded* from warnings.

# Each pair is stored as a set of two strings (e.g., {"a", "a_1"}), and comparison
# is done using set equality, i.e., order does not matter.

# These typically include:
# - semantically equivalent alternatives (e.g., {"conv_time_series", "time_series"})
# - mirrored structures (e.g., {"inhib_a", "inhib_b"})
# - systematic naming conventions (e.g., {"basis1", "basis2"})
# - commonly used argument patterns (e.g., {"args", "kwargs"})
VALID_PAIRS = [
    {"conv_time_series", "time_series"},
    {"inhib_a", "inhib_b"},
    {"excit_a", "excit_b"},
    *(
        {a, b}
        for (a, b) in itertools.combinations(
            ["pytree", "pytree_1", "pytree_2", "pytree_x", "pytree_y"], r=2
        )
    ),
    {"args", "kwargs"},
    *({a, b} for (a, b) in itertools.combinations(["basis", "basis1", "basis2"], r=2)),
    *({a, b} for (a, b) in itertools.combinations(["axis", "axis_1", "axis_2"], r=2)),
    *(
        {a, b}
        for (a, b) in itertools.combinations(
            ["array", "array_1", "array_2", "arrays"], r=2
        )
    ),
]


def collect_similar_parameter_names(
    package,
    root_name: Optional[str] = None,
    similarity_cutoff=0.8,
    valid_pairs: Optional[set[str]] = None,
):
    """
    Recursively collect and group similar parameter names from functions and methods.

    This function traverses the given package and its submodules, extracting parameter
    names from all user-defined functions and methods. Parameter names that are
    lexically similar (based on difflib.get_close_matches) are grouped together.
    This can be used to detect inconsistent naming conventions across a codebase.

    Parameters
    ----------
    package : module
        The root package to analyze (e.g., pynapple).
    root_name : str, optional
        The dotted name of the root package. If not provided, it is inferred from
        package.__name__.
    similarity_cutoff : float, optional
        Similarity threshold between 0 and 1 used to group parameters based on
        lexical similarity.
    valid_pairs :
        Pairs of similar strings that are allowed as distinct parameter names.

    Returns
    -------
    dict
        A dictionary mapping canonical parameter names to a list of tuples.
        Each tuple contains:
            - The actual parameter name
            - The fully qualified function or method path where it appears

        Example
        -------
        {
            "time": [("time", "pynapple.core.Tsd.__init__"), ("t", "pynapple.io.load")],
            ...
        }
    """
    if root_name is None:
        root_name = package.__name__

    if valid_pairs is None:
        valid_pairs = VALID_PAIRS

    results = {}
    visited_ids = set()

    def process_function(func, path):
        if "jaxopt." in path:
            return
        try:
            sig = inspect.signature(func)
            param_names = list(sig.parameters)
            for par in param_names:
                if par in results:
                    results[par].append((par, path))
                    continue  # exact name already exists store
                match = difflib.get_close_matches(
                    par, results.keys(), n=1, cutoff=similarity_cutoff
                )
                if match and not {match[0], par} in VALID_PAIRS:
                    results[match[0]].append((par, path))
                else:
                    results[par] = [(par, path)]
        except Exception:
            pass  # some built-ins or extension modules may not support signature()

    def walk(obj, path_prefix=""):
        if id(obj) in visited_ids:
            return
        visited_ids.add(id(obj))

        if inspect.isfunction(obj) or inspect.ismethod(obj):
            if getattr(obj, "__module__", "").startswith(root_name):
                process_function(obj, path_prefix)

        elif inspect.isclass(obj):
            if getattr(obj, "__module__", "").startswith(root_name):
                for name, member in inspect.getmembers(obj):
                    if name.startswith("_"):
                        continue
                    walk(member, f"{path_prefix}.{name}")

        elif isinstance(obj, types.ModuleType):
            if not getattr(obj, "__name__", "").startswith(root_name):
                return  # external module, skip
            for name, member in inspect.getmembers(obj):
                if name.startswith("_"):
                    continue
                walk(member, f"{path_prefix}.{name}")

    walk(package, root_name)
    return results


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser(
        description="Detect similar but inconsistent parameter names across a package."
    )
    parser.add_argument(
        "--package",
        "-p",
        type=str,
        default="nemos",
        help="Importable Python package to check (e.g., 'nemos', 'torch', 'my_module').",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Similarity threshold (between 0 and 1) for grouping parameter names (default: 0.8)",
    )
    args = parser.parse_args()

    package = args.package
    pkg = importlib.import_module(package)

    logger = logging.getLogger("check_parameter_naming")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    params = collect_similar_parameter_names(pkg, similarity_cutoff=args.threshold)

    for name, occurrences in list(params.items()):
        if all(o == name for o, _ in occurrences):
            params.pop(name)

    if params:
        msg_lines = ["Inconsistency in parameter naming found!\n"]
        for name, occurrences in params.items():
            msg_lines.append(f"{name}:\n")
            for param_name, path in occurrences:
                msg_lines.append(f"\t- {path}: {param_name}\n")
            msg_lines.append("\n")

        logger.warning("".join(msg_lines))
        sys.exit(1)
    else:
        logger.info("No parameter naming inconsistencies found.")
