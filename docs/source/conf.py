# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/main/usage/configuration.html#project-information

import toml
import time
fname = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
project = toml.load(fname)["tool"]["poetry"]["name"]
version = toml.load(fname)["tool"]["poetry"]["version"]
authors = toml.load(fname)["tool"]["poetry"]["authors"][0]
year = time.strftime("%Y")  # current year
year1 = "2023"  # year of first release
year = f"{year1} - {year}" if int(year1) < int(year) else year1
copyright = f"{year}, {authors}"
author = authors
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/main/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.autodoc',  # Include documentation from docstrings
    'sphinx.ext.autosummary',  # todo: Generate autodoc summaries
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.autosectionlabel',  # Allow reference sections using its title
    'sphinx_design',  # designing beautiful, view size responsive web components; enables sphinx_rtd_theme (see below)
    'sphinx_copybutton',  # add a little “copy” button to the right of your code blocks
]

# autodoc_default_flags = ["show-inheritance"]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Enable numref
numfig = True
numfig_secnum_depth = 1  # applies only if section numbering is activated via the :numbered: option of the toctree directive

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

html_theme = 'sphinx_rtd_theme'  # 'alabaster'
html_static_path = ['_static']
