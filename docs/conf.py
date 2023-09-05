# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/main/usage/configuration.html#project-information

import toml
import time
fname = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
name = toml.load(fname)["tool"]["poetry"]["name"]
version = toml.load(fname)["tool"]["poetry"]["version"]
authors = toml.load(fname)["tool"]["poetry"]["authors"][0]
year = time.strftime("%Y")

project = name
copyright = f"{year}, {authors}"
author = authors
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/main/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',  # todo: test this
    'sphinx.ext.coverage',  # todo: test this
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',
    'sphinx.ext.autosectionlabel',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/main/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # 'alabaster'
html_static_path = ['_static']
