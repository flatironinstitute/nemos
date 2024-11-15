# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import nemos
import sys, os
from pathlib import Path

# conf.py
# def strip_class_name_from_methods(app, what, name, obj, options, signature, return_annotation):
#     # Check if it’s a method in a class and remove the class prefix
#     if what == "method" and "." in name:
#         # Only use the method part of the full name
#         method_name = name.split(".")[-1]
#
#         # Emit the updated signature and return annotation without class name
#         return method_name + signature, return_annotation
#     return signature, return_annotation
#
# def setup(app):
#     app.connect("autodoc-process-signature", strip_class_name_from_methods)


sys.path.insert(0, str(Path('..', 'src').resolve()))
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
    #'numpydoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',  # Links to source code
    'sphinx.ext.doctest',
    'sphinx_copybutton',  # Adds copy button to code blocks
    'sphinx_design',  # For layout components
    'myst_nb',
    'sphinx_contributors',
    'sphinx_code_tabs',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_togglebutton',
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

autodoc_typehints = "description"  # Use "description" to place hints in the description
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "NDArray": "NDArray",
    "TsdFrame": "pynapple.TsdFrame",
    "JaxArray": "JaxArray",
}

numfig = True

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
    "navigation_depth": 3,
}

html_context = {
    "default_mode": "light",
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
copybutton_prompt_text = r">>> |\$ |# "
copybutton_prompt_is_regexp = True

nitpicky = True
