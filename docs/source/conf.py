# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TemporalVAE'
copyright = '2025, Yijun Liu; Yuanhua Huang'
author = 'Yijun Liu; Yuanhua Huang'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.bibtex",
    # "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
# The theme to use for HTML and HTML Help pages.  See the documentation for
# notebooks = []
# notebook = [
#     "VelocityBasics.ipynb",
#     "DynamicalModeling.ipynb",
#     "DifferentialKinetics.ipynb",
# ]
# notebooks.extend(notebook)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

import os
# import django
import sys
# sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..')) #更改成这个路径
