# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import logging
from sphinx.ext.autodoc import between


sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../PyLorentz/'))
# from PyLorentz.dataset import base_dataset, defocused_dataset, through_focal_series
# from PyLorentz.utils import Microscope


# -- Project information -----------------------------------------------------

project = "PyLorentz"
copyright = "2018, UChicago Argonne, LLC"
author = "Arthur McCray, Tim Cote, CD Phatak"

# The full version, including alpha/beta/rc tags
release = "2.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # "sphinx.ext.autosectionlabel",
]

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "numba",
    "ipympl",
    "jupyter",
    "scikit-image",
    "matplotlib",
    "ncempy",
    "colorcet",
    "black",
    "tqdm",
    "pytorch",
    "cupy",
    "ncempy",
    "skimage",
    "colormap",
    "physcon",
    "ipywidgets",
    "tifffile",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

master_doc = "index"


autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
}


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)