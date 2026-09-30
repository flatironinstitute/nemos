"""Draw the map of ``nemos.solvers._second_order`` used by the developer notes.

Two SVGs are written to ``docs/assets``: a light one and a ``_dark`` companion, the
pair the docs switch between with ``:class: only-light`` / ``:class: only-dark``.

The layers are the runtime import graph, which this script asserts against the source
rather than trusting the drawing: a module may only import from a layer below its own,
and imports made solely for annotations do not count.

Usage:
    python scripts/generate_second_order_module_map.py
"""

import ast
import pathlib
import subprocess

MODULE = pathlib.Path("src/nemos/solvers/_second_order")
ASSETS = pathlib.Path("docs/assets")

# A module may import only from something earlier in this order: an earlier layer, or,
# within a layer, an earlier entry. The bases share the solvers' layer rather than
# sitting under it -- a base is what a solver *is*, not something it depends on -- and
# come first within it, which is what makes the order total.
LAYERS = [
    ("shared vocabulary", ["_typing", "_utils"]),
    (
        "the four things a solver assembles",
        ["_curvature", "_direction", "_linesearches", "_loop"],
    ),
    (
        "solvers, and the bases they are built on",
        ["_base", "_hessian_mixins", "_newton", "_lbfgs"],
    ),
]

# Drawn on the solvers' rank, but styled as bases rather than as dependencies.
BASES = {"_base", "_hessian_mixins"}


def contents(mod: str) -> list[str]:
    """What a module defines, read off the source so the drawing cannot drift.

    Classes for the modules that hold them; type variables and functions for the two
    that hold neither.
    """
    tree = ast.parse((MODULE / f"{mod}.py").read_text())
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    if classes:
        return classes
    names = [
        n.targets[0].id
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", "") == "TypeVar"
    ]
    aliases = [
        n.targets[0].id
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and not isinstance(n.value, ast.Call)
    ]
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return ([", ".join(names)] if names else []) + aliases + funcs


THEMES = {
    "": dict(
        bg="none",
        fg="#1a1a1a",
        muted="#6b6b6b",
        edge="#8a8a8a",
        fills=["#f2f2f2", "#e3edf7", "#e2f0e6"],
        strokes=["#b8b8b8", "#7ba3cc", "#74ab89"],
        base_fill="#ece4f5",
        base_stroke="#9b83c4",
    ),
    "_dark": dict(
        bg="none",
        fg="#e8e8e8",
        muted="#9a9a9a",
        edge="#6f6f6f",
        fills=["#2a2a2a", "#243642", "#24382c"],
        strokes=["#5a5a5a", "#5b8bb8", "#5c9173"],
        base_fill="#2e2740",
        base_stroke="#8069ab",
    ),
}


def runtime_imports() -> dict[str, set[str]]:
    """Intra-module imports that survive at runtime, i.e. not under ``TYPE_CHECKING``."""
    mods = {p.stem for p in MODULE.glob("*.py")}
    graph = {}
    for path in sorted(MODULE.glob("*.py")):
        if path.stem == "__init__":
            continue
        tree = ast.parse(path.read_text())
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
                guarded.update(id(child) for child in ast.walk(node))
        graph[path.stem] = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module in mods
            and id(node) not in guarded
        }
    return graph


def check_layering(graph: dict[str, set[str]]) -> None:
    """Assert the order the drawing claims is the order the source actually has."""
    order = {
        m: (i, j) for i, (_, mods) in enumerate(LAYERS) for j, m in enumerate(mods)
    }
    missing = set(graph) - set(order)
    if missing:
        raise SystemExit(f"module(s) not placed in a layer: {sorted(missing)}")
    for mod, deps in graph.items():
        for dep in deps:
            if order[dep] >= order[mod]:
                raise SystemExit(
                    f"{mod} {order[mod]} imports {dep} {order[dep]}: "
                    "the drawing claims an order the code does not have"
                )


def inheritance() -> set[tuple[str, str]]:
    """``(subclass module, base module)`` for every base imported from this package."""
    mods = {p.stem for p in MODULE.glob("*.py")}
    edges = set()
    for path in sorted(MODULE.glob("*.py")):
        tree = ast.parse(path.read_text())
        origin = {
            alias.asname or alias.name: node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module in mods
            for alias in node.names
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    root = base
                    while isinstance(root, ast.Subscript):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in origin:
                        edges.add((path.stem, origin[root.id]))
    return edges


def node_label(mod: str) -> str:
    rows = "".join(
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">{name}</FONT></TD></TR>'
        for name in contents(mod)
    )
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="1">'
        f'<TR><TD ALIGN="LEFT"><B>{mod}</B></TD></TR>{rows}</TABLE>>'
    )


def dot_source(graph: dict[str, set[str]], theme: dict) -> str:
    """One rank per layer, top to bottom, with the bases beside the solvers."""
    order = {
        m: (i, j) for i, (_, mods) in enumerate(LAYERS) for j, m in enumerate(mods)
    }
    inherits = inheritance()
    out = [
        "digraph second_order {",
        f'  bgcolor="{theme["bg"]}";',
        "  rankdir=TB; splines=spline; nodesep=0.45; ranksep=1.0;",
        '  graph [fontname="Helvetica,Arial,sans-serif"];',
        '  node [shape=box, style="rounded,filled", penwidth=1.3, margin="0.16,0.10",'
        f'        fontname="Helvetica,Arial,sans-serif", fontcolor="{theme["fg"]}"];',
        f'  edge [color="{theme["edge"]}", arrowsize=0.7, penwidth=1.1];',
    ]
    for i, (title, mods) in reversed(list(enumerate(LAYERS))):
        out.append("  { rank=same;")
        out.append(
            f'    "L{i}" [label="{title}", shape=plaintext, style="", '
            f'fontsize=10, fontcolor="{theme["muted"]}"];'
        )
        for mod in mods:
            base = mod in BASES
            fill = theme["base_fill"] if base else theme["fills"][i]
            stroke = theme["base_stroke"] if base else theme["strokes"][i]
            dash = ', style="rounded,filled,dashed"' if base else ""
            out.append(
                f'    "{mod}" [label={node_label(mod)}, '
                f'fillcolor="{fill}", color="{stroke}"{dash}];'
            )
        out.append("  }")
    spine = " -> ".join(f'"L{i}"' for i in reversed(range(len(LAYERS))))
    out.append(f"  {spine} [style=invis];")
    for i, (_, mods) in enumerate(LAYERS):
        out.append(f'  "L{i}" -> "{mods[0]}" [style=invis];')
    # keep the bases to the left of the solvers on their shared rank
    for a, b in zip(LAYERS[-1][1], LAYERS[-1][1][1:]):
        out.append(f'  "{a}" -> "{b}" [style=invis, weight=10];')

    for mod, deps in sorted(graph.items()):
        for dep in sorted(deps):
            if dep == "_typing":  # every module imports it; drawing it buries the rest
                continue
            if (mod, dep) in inherits:
                out.append(
                    f'  "{mod}" -> "{dep}" [arrowhead=onormal, arrowsize=1.0, '
                    f'constraint=false, color="{theme["base_stroke"]}"];'
                )
            else:
                same = order[mod][0] == order[dep][0]
                out.append(
                    f'  "{mod}" -> "{dep}"{" [constraint=false]" if same else ""};'
                )
    out.append(
        f'  label=<<BR/><FONT POINT-SIZE="10" COLOR="{theme["muted"]}">'
        "a plain arrow points from a module to one it imports at runtime; a hollow one "
        "from a solver to a base it inherits. Imports made only for annotations are not "
        "runtime imports and are not drawn, and neither is <B>_typing</B>, which every "
        "module uses."
        "</FONT>>; labelloc=b;"
    )
    out.append("}")
    return "\n".join(out)


def main() -> None:
    graph = runtime_imports()
    check_layering(graph)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for suffix, theme in THEMES.items():
        target = ASSETS / f"second_order_module{suffix}.svg"
        subprocess.run(
            ["dot", "-Tsvg", "-o", str(target)],
            input=dot_source(graph, theme),
            text=True,
            check=True,
        )
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
