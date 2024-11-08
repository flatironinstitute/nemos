# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import nemos
import sys, os

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('sphinxext'))


project = 'nemos'
copyright = '2024, SJ Venditto'
author = 'SJ Venditto'
version = release = nemos.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The Root document
root_doc = "index"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',  # Links to source code
    'sphinx.ext.doctest',
    'sphinx_copybutton',  # Adds copy button to code blocks
    'sphinx_design',  # For layout components
    'myst_nb',
    'sphinx_contributors'
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    # "deflist",
    "dollarmath",
    # "fieldlist",
    "html_admonition",
    "html_image",
    # "replacements",
    # "smartquotes",
    # "strikethrough",
    # "substitution",
    # "tasklist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', 'nextgen', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_material'
html_static_path = ['_static']

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = True
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'show-inheritance': True,
    }


html_theme = 'pydata_sphinx_theme'

html_logo = "assets/NeMoS_Logo_CMYK_Full.svg"
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
    "header_links_before_dropdown": 5,
}

html_context = {
    "default_mode": "light",
}

# Path for static files (custom stylesheets or JavaScript)
html_static_path = ['assets/stylesheets']
html_css_files = ['custom.css']

# Copybutton settings (to hide prompt)
copybutton_prompt_text = r">>> |\$ |# "
copybutton_prompt_is_regexp = True

# Enable markdown and notebook support
myst_enable_extensions = ["colon_fence"]  # For improved markdown

# # ----------------------------------------------------------------------------
# # -- Autodoc and Napoleon Options -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
napoleon_numpy_docstring = True

nitpicky = True
