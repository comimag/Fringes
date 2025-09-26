# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/main/usage/configuration.html#project-information

import time

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# todo: use importlib.metadata
fname = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
with open(fname, "rb") as f:
    data = tomllib.load(f)
    name = data["project"]["name"]
    project = name
    version = data["project"]["version"]
    authors = data["project"]["authors"][0]["name"]
year = time.strftime("%Y")  # current year
year0 = "2023"  # year of first release
year = f"{year0} - {year}" if int(year0) < int(year) else year0
copyright = f"{year}, {authors}"
author = authors
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/main/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.autosummary",  # todo: Generate autodoc summaries
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.autosectionlabel",  # Allow reference sections using its title
    "sphinx_design",  # designing beautiful, view size responsive web components; enables sphinx_rtd_theme (see below)
    "sphinx_copybutton",  # add a little “copy” button to the right of your code blocks
]

# autodoc_default_flags = ["show-inheritance"]
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Enable numref
numfig = True
numfig_secnum_depth = (
    1  # applies only if section numbering is activated via the :numbered: option of the toctree directive
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

html_theme = "sphinx_rtd_theme"  # "alabaster"
html_static_path = ["_static"]
