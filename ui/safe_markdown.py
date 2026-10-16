"""Escape dollar signs in Streamlit markdown so currency never renders as LaTeX.

Streamlit treats text between two ``$`` characters as a math expression, so a
message such as "need $2,250 more ... the $3,000 target" would be mangled. The
application never uses LaTeX, so every markdown-rendering element is wrapped
once at start-up to escape unescaped ``$``.
"""

from __future__ import annotations

import functools
import re

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")
ELEMENTS = ("markdown", "caption", "info", "warning", "error", "success", "toast")
_installed = False


def escape_dollars(body: object) -> object:
    return _UNESCAPED_DOLLAR.sub(r"\\$", body) if isinstance(body, str) else body


def install() -> None:
    global _installed
    if _installed:
        return
    for name in ELEMENTS:
        cls_orig = getattr(DeltaGenerator, name)

        @functools.wraps(cls_orig)
        def method(self, body, *args, __orig=cls_orig, **kwargs):
            return __orig(self, escape_dollars(body), *args, **kwargs)

        setattr(DeltaGenerator, name, method)
        mod_orig = getattr(st, name)

        @functools.wraps(mod_orig)
        def function(body, *args, __orig=mod_orig, **kwargs):
            return __orig(escape_dollars(body), *args, **kwargs)

        setattr(st, name, function)
    _installed = True
