"""Sphinx configuration for the msutils documentation.

Autodoc imports the ``msutils`` package, so the build environment must have it
and python-casacore installed (``uv sync --group docs --all-extras`` locally;
Read the Docs installs them via ``.readthedocs.yaml`` / ``docs/requirements.txt``).
The package lives under ``src/``, added to sys.path below so an
editable/uninstalled checkout also builds.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.abspath("../src"))

from msutils import __version__

# -- Project information -----------------------------------------------------

project = "msutils"
author = "Sphesihle Makhathini"
copyright = f"{datetime.now(tz=UTC).year}, {author}"

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Warnings are build-relevant but do not fail the build on autodoc targets
# that need an optional extra installed.
nitpicky = False

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Deliberately no `"members": True` default. The package root re-exports
# everything, so a bare `automodule:: msutils` would document each definition a
# second time and Sphinx would flag every one as a duplicate object. Each
# directive in docs/api/index.rst names its members explicitly instead, which
# documents every definition exactly once, at the module that defines it.
autodoc_default_options = {
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_inherit_docstrings = False

# The optional stacks are not installed on every docs builder, and autodoc
# imports what it documents. Mocking them keeps the API pages complete without
# making the whole build depend on xradio/africanus resolving.
autodoc_mock_imports = ["africanus", "xradio", "xarray", "xarray_ms", "matplotlib"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
# Fetching remote inventories is the one part of the build that needs network.
# Without a timeout an offline build hangs on it instead of degrading to a
# warning, which is a poor experience for `uv run sphinx-build` on a laptop.
intersphinx_timeout = 10

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"msutils {release}"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/shinobi-dosho/msutils/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- MyST (markdown) ---------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
