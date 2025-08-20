# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'graphcalc'
copyright = '2024, Randy Davila, PhD'
author = 'Randy Davila, PhD'
release = '1.2.12'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Enable Sphinx extensions for autodoc and better docstring support
extensions = [
    'sphinx.ext.autodoc',      # Automatically document members from docstrings
    'sphinx.ext.napoleon',     # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',     # Adds links to the source code in the documentation
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------
# Automatically include module/class docstrings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Options for HTML output -------------------------------------------------
# Use the Read the Docs theme (or alabaster if preferred)
html_theme = 'sphinx_rtd_theme'  # Use 'alabaster' or 'sphinx_rtd_theme'
# html_static_path = ['_static']

# -- Add paths to sys.path for autodoc to locate graphcalc -------------------
# This allows Sphinx to locate the 'graphcalc' package for documentation

import os
import sys

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath('../../src'))
# sys.path.insert(0, os.path.abspath('../../'))  # Adjust path to locate 'graphcalc'
