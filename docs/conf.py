# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import typing
import urllib.request
from importlib.metadata import version
from pathlib import Path

release: str = version("nemos")
# this will grab major.minor.patch (excluding any .devN afterwards, which should only
# show up when building locally during development)
version: str = ".".join(release.split(".")[:3])

sys.path.insert(0, str(Path("..", "src").resolve()))
sys.path.insert(0, os.path.abspath("sphinxext"))


project = "nemos"
copyright = "2024"
author = "E Balzani"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The Root document
root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "nemos_autodoc_skip_member",  # skip custom members from autodoc
    # Prioritize custom logic by listing just after autodoc.
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",  # Links to source code
    "sphinx.ext.doctest",
    "sphinx_copybutton",  # Adds copy button to code blocks
    "sphinx_design",  # For layout components
    "myst_nb",
    "sphinx_contributors",
    "sphinxcontrib.bibtex",
    "sphinx_code_tabs",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_togglebutton",
    "matplotlib.sphinxext.plot_directive",
    "matplotlib.sphinxext.mathmpl",
    "sphinx.ext.intersphinx",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "dollarmath",
    "html_admonition",
    "html_image",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "docstrings", "Thumbs.db", "nextgen", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": " __add__, __mul__, __pow__",
}

# # napolean configs
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# SPHINXCONTRIB-BIBTEX
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

autodoc_typehints = "description"  # Use "description" to place hints in the description
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "NDArray": "NDArray",
    "TsdFrame": "pynapple.TsdFrame",
    "JaxArray": "JaxArray",
}
autodoc_typehints_format = "short"

numfig = True

html_theme = "pydata_sphinx_theme"

html_favicon = "assets/NeMoS_favicon.ico"

# Additional theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/flatironinstitute/nemos/",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "X",
            "url": "https://x.com/nemos_neuro",
            "icon": "fab fa-square-x-twitter",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": True,
    "header_links_before_dropdown": 6,
    "navigation_depth": 3,
    "logo": {
        "image_light": "_static/NeMoS_Logo_CMYK_Full.svg",
        "image_dark": "_static/NeMoS_Logo_CMYK_White.svg",
    },
    "secondary_sidebar_items": {
        "[!a]?[!p]?[!i]**": ["page-toc", "sourcelink"],
        "background/basis/README": [],
    },
}

html_sidebars = {
    "index": [],
    "installation": [],
    "quickstart": [],
    "benchmarking": [],
    "background/README": [],
    "how_to_guide/README": [],
    "tutorials/README": [],
    "**": ["search-field.html", "sidebar-nav-bs.html"],
}


# Path for static files (custom stylesheets or JavaScript)
html_static_path = ["assets/stylesheets", "assets", "javascripts"]
html_css_files = ["custom.css"]

html_js_files = ["https://code.iconify.design/2/2.2.1/iconify.min.js"]

# Copybutton settings (to hide prompt)
# Exclude prompts/output via Pygments CSS classes rather than a text regex.
# `.gp` covers both `>>>` and `...` continuation prompts (and shell `$`),
# `.go` covers console output. This avoids dropping `...` continuation lines.
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#automatic-exclusion-of-prompts-from-the-copies
copybutton_exclude = ".linenos, .gp, .go"

sphinxemoji_style = "twemoji"

nb_execution_timeout = 60 * 15  # Set timeout in seconds (e.g., 15 minutes)

nitpicky = True

# Get exclusion patterns from an environment variable
exclude_tutorials = os.environ.get("EXCLUDE_TUTORIALS", "false").lower() == "true"

if exclude_tutorials:
    # myst-nb matches these with PurePosixPath.match, which anchors from the right
    # and treats "**" as non-recursive, so a glob covers a single directory level.
    # Listing the files instead keeps notebooks in new subdirectories excluded
    # without anyone having to add a deeper pattern by hand.
    _docs_root = Path(__file__).parent
    nb_execution_excludepatterns = [
        path.relative_to(_docs_root).as_posix()
        for root in ("tutorials", "how_to_guide", "background")
        for path in sorted((_docs_root / root).rglob("*.md"))
    ]

viewcode_follow_imported_members = True

# option for mpl extension
plot_html_show_formats = False

# raise an error if exec error in notebooks
nb_execution_raise_on_error = True

# cache notebooks when possible
nb_execution_mode = "cache"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

full_api = os.environ.get("NEMOS_FULL_API", "false").lower() == "true"

# ---- API index generation ----
api_order = [
    "glm.rst",
    "glm_hmm.rst",
    "basis.rst",
    "observation_models.rst",
    "regularizers.rst",
    "io.rst",
    "solvers.rst",
    "stochastic_optimization.rst",
    "convolve.rst",
    "simulations.rst",
    "identifiability.rst",
]
api_dir = Path("api")
# API index page (api/index.rst) is auto-generated. It starts with a hidden toctree
# including all the rst pages above, and then includes their text (in order), without
# the sphinx anchor and the toctree argument to the autosummary directive
api_index = """.. _api:

API Reference
=============

.. toctree::
   :hidden:

"""
api_index += "   "
api_index += "\n   ".join(mod.replace(".rst", "") for mod in api_order)
api_index += "\n"

for api_rst in api_order:
    api_rst = api_dir / api_rst
    contents = api_rst.read_text().split("\n")
    # two lines we want to throw away: the sphinx anchor (e.g., ".. _synthesis-api") and
    # the line that tells autosummary to create a toctree (e.g., ":toctree: generated")
    contents = [
        c
        for c in contents
        if not c.strip().startswith(".. _") and not c.strip().startswith(":toctree:")
    ]
    api_index += "\n".join(contents)

if full_api:
    autodoc_default_options["private-members"] = True
    nitpicky = False

    # A handful of modules (utils, observation_models, regularizer) define a
    # module-level ``__dir__`` returning ``__all__``. autodoc discovers module
    # members through ``dir()``, so those modules would expose only their public
    # names no matter what options we pass -- ``:ignore-module-all:`` does not
    # help, since it is ``__all__`` via ``__dir__`` that hides them. Import the
    # package eagerly and drop the overrides so ``dir()`` falls back to the
    # module ``__dict__``. Safe here because no nemos module pairs ``__dir__``
    # with a lazy ``__getattr__``.
    import importlib
    import pkgutil

    import nemos

    for _mod_info in pkgutil.walk_packages(nemos.__path__, prefix="nemos."):
        try:
            importlib.import_module(_mod_info.name)
        except ImportError as e:
            print(f"full API: could not import {_mod_info.name}: {e}")
    for _mod in list(sys.modules.values()):
        if getattr(_mod, "__name__", "").startswith("nemos") and "__dir__" in vars(
            _mod
        ):
            del _mod.__dir__

    # add full API page without stripping toctree
    api_rst = api_dir / "full.rst"
    contents = api_rst.read_text().split("\n")
    api_index += "\n".join(contents)

else:
    exclude_patterns += ["api/full.rst", "api/generated/full/**"]


(api_dir / "index.rst").write_text(api_index)

# ---- Download admonition for runnable notebook docs ----
# Every jupytext MyST .md doc is written to _build/jupyter_execute/<doc>.ipynb
# by myst_nb on each build, and the {nb-download} role links to that generated
# notebook. We inject the admonition just after the jupytext frontmatter so all
# runnable tutorials/how-to/background pages get a download link automatically,
# without editing the source files.
_NB_DOC_ROOTS = ("tutorials/", "how_to_guide/", "background/")


def add_download_admonition(app, docname, source):
    if not (docname.startswith(_NB_DOC_ROOTS) or docname == "quickstart"):
        return
    lines = source[0].splitlines(keepends=True)
    # require a jupytext frontmatter block (fenced by ---) to skip plain .md
    # pages such as the README index files living in these directories
    if not lines or lines[0].strip() != "---":
        return
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None or "jupytext" not in "".join(lines[1:end]):
        return
    stem = docname.split("/")[-1]
    admonition = (
        "\n"
        ":::{admonition} Download\n"
        ":class: important\n"
        "\n"
        f"Download this notebook: **{{nb-download}}`{stem}.ipynb`**!\n"
        "\n"
        ":::\n"
        "\n"
    )
    lines.insert(end + 1, admonition)
    source[0] = "".join(lines)


def strip_generic_bases(app, name, obj, options, bases):
    """Render ``Base[...]`` as bare ``Base`` in the Bases: line (drops generic clutter)."""
    for i, base in enumerate(bases):
        origin = typing.get_origin(base)
        if origin is not None:
            bases[i] = origin


# Download the latest benchmark summary so the DataTable can be served same-origin.
_benchmark_csv_url = "https://users.flatironinstitute.org/~ebalzani/nemos/benchmark/aggregate_summary.csv"
_benchmark_csv_dst = Path("assets") / "aggregate_summary.csv"
try:
    urllib.request.urlretrieve(_benchmark_csv_url, _benchmark_csv_dst)
    print(f"Downloaded benchmark summary -> {_benchmark_csv_dst}")
except Exception as e:
    raise RuntimeError(
        f"Could not fetch benchmark summary from {_benchmark_csv_url}."
    ) from e


def _add_benchmark_assets(app, pagename, templatename, context, doctree):
    if pagename != "benchmarking":
        return
    app.add_css_file(
        "https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css"
    )
    app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
    app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
    app.add_js_file(
        "https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"
    )
    app.add_js_file("https://cdn.plot.ly/plotly-2.35.2.min.js")
    app.add_js_file("benchmark-table.js")


def setup(app):
    app.connect("source-read", add_download_admonition)
    app.connect("autodoc-process-bases", strip_generic_bases)
    app.connect("html-page-context", _add_benchmark_assets)
