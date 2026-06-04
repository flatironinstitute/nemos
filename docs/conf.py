# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os
import typing
from pathlib import Path

from importlib.metadata import version
release: str = version("nemos")
# this will grab major.minor.patch (excluding any .devN afterwards, which should only
# show up when building locally during development)
version: str = ".".join(release.split('.')[:3])

sys.path.insert(0, str(Path('..', 'src').resolve()))
sys.path.insert(0, os.path.abspath('sphinxext'))


project = 'nemos'
copyright = '2024'
author = 'E Balzani'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The Root document
root_doc = "index"

extensions = [
    'sphinx.ext.autodoc',
    'nemos_autodoc_skip_member',  # skip custom members from autodoc
                                  # Prioritize custom logic by listing just after autodoc.
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',  # Links to source code
    'sphinx.ext.doctest',
    'sphinx_copybutton',  # Adds copy button to code blocks
    'sphinx_design',  # For layout components
    'myst_nb',
    'sphinx_contributors',
    "sphinxcontrib.bibtex",
    'sphinx_code_tabs',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_togglebutton',
    'matplotlib.sphinxext.plot_directive',
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

templates_path = ['_templates']
exclude_patterns = ['_build', "docstrings", 'Thumbs.db', 'nextgen', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = True
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': ' __add__, __mul__, __pow__'
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

html_theme = 'pydata_sphinx_theme'

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
    "installation":[],
    "quickstart": [],
    "background/README": [],
    "how_to_guide/README": [],
    "tutorials/README": [],
    "**": ["search-field.html", "sidebar-nav-bs.html"],
}


# Path for static files (custom stylesheets or JavaScript)
html_static_path = ['assets/stylesheets', "assets"]
html_css_files = ['custom.css']

html_js_files = [
    "https://code.iconify.design/2/2.2.1/iconify.min.js"
]

# Copybutton settings (to hide prompt)
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

sphinxemoji_style = 'twemoji'

nb_execution_timeout = 60 * 15  # Set timeout in seconds (e.g., 15 minutes)

nitpicky = True

# Get exclusion patterns from an environment variable
exclude_tutorials = os.environ.get("EXCLUDE_TUTORIALS", "false").lower() == "true"

if exclude_tutorials:
    nb_execution_excludepatterns = ["tutorials/*md", "how_to_guide/*md", "background/*md", "background/*/*md"]

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

# ---- API index generation ----
api_order = [
    "glm.rst",
    "glm_hmm.rst",
    "basis.rst",
    "observation_models.rst",
    "regularizers.rst",
    "io.rst",
    "solvers.rst",
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


def setup(app):
    app.connect("source-read", add_download_admonition)
    app.connect("autodoc-process-bases", strip_generic_bases)
