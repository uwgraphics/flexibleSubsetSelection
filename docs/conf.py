import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'Flexible Subset Selection Examples'
author = 'Connor Bailey'
copyright = '2025, Connor Bailey'
release = '0.2'

# -- Extensions ----------------------------------------------------------------
extensions = [
    'myst_nb', 
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx.ext.mathjax',
    'sphinx_togglebutton',
    'sphinx_design',
    'autoapi.extension'
]
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "amsmath",
]

# -- HTML output ---------------------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = None  
html_title = project

html_theme_options = {
    "repository_url": "https://github.com/uwgraphics/flexibleSubsetSelection",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
}

# -- AutoAPI configuration -----------------------------------------------------
autoapi_type = 'python'
autoapi_dirs = ['../src/flexibleSubsetSelection']
autosummary_generate = True
autoapi_add_toctree_entry = False
autoapi_root = 'api'

# Paths and excludes
templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']
nb_execution_mode = "off"