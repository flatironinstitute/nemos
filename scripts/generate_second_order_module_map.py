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

# Each layer may import only from the layers before it.
LAYERS = [
    ("shared vocabulary", ["_typing", "_utils"]),
    (
        "the four things a solver assembles",
        ["_curvature", "_direction", "_linesearches", "_loop"],
    ),
    ("wiring", ["_base", "_hessian_mixins"]),
    ("solvers", ["_newton", "_lbfgs"]),
]

CONTENTS = {
    "_typing": ["Y, S, R, D, Aux", "HvpFn"],
    "_utils": ["map_blocks"],
    "_curvature": ["AbstractCurvature", "NewtonCurvature", "LBFGSCurvature"],
    "_direction": [
        "AbstractDirection",
        "LinearSolveDirection",
        "ProxQuadraticDirection",
    ],
    "_linesearches": [
        "AbstractLineSearch",
        "ArmijoBacktracking",
        "TsengYunBacktracking",
    ],
    "_loop": ["Loop"],
    "_base": ["AbstractSecondOrderSolver", "SecondOrderState"],
    "_hessian_mixins": ["HessianMixin", "HessianSolverMixin"],
    "_newton": ["BaseNewtonSolver", "Newton", "ProximalNewton"],
    "_lbfgs": ["ProximalLBFGS"],
}

THEMES = {
    "": dict(
        bg="none",
        fg="#1a1a1a",
        muted="#6b6b6b",
        edge="#8a8a8a",
        fills=["#f2f2f2", "#e3edf7", "#ece4f5", "#e2f0e6"],
        strokes=["#b8b8b8", "#7ba3cc", "#9b83c4", "#74ab89"],
    ),
    "_dark": dict(
        bg="none",
        fg="#e8e8e8",
        muted="#9a9a9a",
        edge="#6f6f6f",
        fills=["#2a2a2a", "#243642", "#2e2740", "#24382c"],
        strokes=["#5a5a5a", "#5b8bb8", "#8069ab", "#5c9173"],
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
    rank = {m: i for i, (_, mods) in enumerate(LAYERS) for m in mods}
    missing = set(graph) - set(rank)
    if missing:
        raise SystemExit(f"module(s) not placed in a layer: {sorted(missing)}")
    for mod, deps in graph.items():
        for dep in deps:
            if rank[dep] >= rank[mod]:
                raise SystemExit(
                    f"{mod} (layer {rank[mod]}) imports {dep} (layer {rank[dep]}): "
                    "the drawing claims a layering the code does not have"
                )


def node_label(mod: str) -> str:
    rows = "".join(
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">{name}</FONT></TD></TR>'
        for name in CONTENTS[mod]
    )
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="1">'
        f'<TR><TD ALIGN="LEFT"><B>{mod}</B></TD></TR>{rows}</TABLE>>'
    )


def dot_source(graph: dict[str, set[str]], theme: dict) -> str:
    """One rank per layer, top to bottom, so every arrow points downward."""
    layer_of = {m: i for i, (_, mods) in enumerate(LAYERS) for m in mods}
    out = [
        "digraph second_order {",
        f'  bgcolor="{theme["bg"]}";',
        "  rankdir=TB; splines=spline; nodesep=0.45; ranksep=0.95;",
        '  graph [fontname="Helvetica,Arial,sans-serif"];',
        '  node [shape=box, style="rounded,filled", penwidth=1.3, margin="0.16,0.10",'
        f'        fontname="Helvetica,Arial,sans-serif", fontcolor="{theme["fg"]}"];',
        f'  edge [color="{theme["edge"]}", arrowsize=0.7, penwidth=1.1];',
    ]
    # top to bottom: solvers first, foundation last
    for i, (title, mods) in reversed(list(enumerate(LAYERS))):
        out.append("  { rank=same;")
        out.append(
            f'    "L{i}" [label="{title}", shape=plaintext, style="", '
            f'fontsize=10, fontcolor="{theme["muted"]}"];'
        )
        for mod in mods:
            out.append(
                f'    "{mod}" [label={node_label(mod)}, '
                f'fillcolor="{theme["fills"][i]}", color="{theme["strokes"][i]}"];'
            )
        out.append("  }")
    # invisible spine keeps the layers stacked and the labels in their gutter
    spine = " -> ".join(f'"L{i}"' for i in reversed(range(len(LAYERS))))
    out.append(f"  {spine} [style=invis];")
    for i, (_, mods) in enumerate(LAYERS):
        out.append(f'  "L{i}" -> "{mods[0]}" [style=invis];')

    for mod, deps in sorted(graph.items()):
        for dep in sorted(deps):
            if dep == "_typing":  # every module imports it; drawing it buries the rest
                continue
            same = layer_of[mod] - layer_of[dep] > 1
            out.append(f'  "{mod}" -> "{dep}"{" [constraint=false]" if same else ""};')
    out.append(
        f'  label=<<BR/><FONT POINT-SIZE="10" COLOR="{theme["muted"]}">'
        "an arrow points from a module to one it imports at runtime; imports made only "
        "for annotations are not runtime imports and are not drawn, and neither is "
        "<B>_typing</B>, which every module uses"
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
