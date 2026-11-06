import ast
import inspect
import textwrap
from collections import defaultdict


def public_callers_of(cls, attr_name, target_methods):
    """Map each public method of `outer_cls` to the methods in `target_methods`
    it reaches on `self.<attr_name>`, directly or through other self methods."""

    classes = [c for c in cls.__mro__ if c.__module__.startswith("nemos.")]
    direct_targets = defaultdict(
        set
    )  # method name -> targets it calls itself (any class in the MRO)
    self_refs = defaultdict(
        set
    )  # method name -> self.<name> lookups (any class in the MRO)
    for outer_cls in classes:
        source = textwrap.dedent(inspect.getsource(outer_cls))
        tree = ast.parse(source)
        class_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))

        for func_node in class_node.body:
            # skip class attributes, nested classes, etc.
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            found_targets, found_self_refs = set(), set()
            for node in ast.walk(func_node):
                if not isinstance(node, ast.Attribute):
                    continue
                owner = node.value  # the part before the last dot

                # self.<attr_name>.<method>: a call into the wrapped instance
                if (
                    isinstance(owner, ast.Attribute)
                    and owner.attr == attr_name
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id == "self"
                    and node.attr in target_methods
                ):
                    found_targets.add(node.attr)

                # self.<name>: maybe another method; non-methods are dropped in the walk
                elif isinstance(owner, ast.Name) and owner.id == "self":
                    found_self_refs.add(node.attr)

            direct_targets[func_node.name] |= found_targets
            self_refs[func_node.name] |= found_self_refs

    # keys: outer-class methods (public and private)
    # values: set of MyClass methods each one ends up calling, directly or through other outer-class methods
    reachable = {}

    def collect_targets(method_name, visited):
        if method_name in reachable:
            return reachable[method_name]
        targets = set(direct_targets.get(method_name, ()))
        for ref in self_refs.get(method_name, ()):
            # follow only methods defined in the class, and don't loop on recursion
            if ref in direct_targets and ref not in visited:
                targets |= collect_targets(ref, visited | {ref})
        return targets

    for method_name in direct_targets:
        reachable[method_name] = collect_targets(method_name, {method_name})

    result = defaultdict(set)
    result.update(
        {
            name: targets
            for name, targets in reachable.items()
            if targets and not name.startswith("_")
        }
    )
    return result


if __name__ == "__main__":
    import nemos as nmo

    targets = {"log_likelihood", "_negative_log_likelihood"}
    PATHS = public_callers_of(nmo.glm.PopulationGLM, "_observation_model", targets)
    print(PATHS)
