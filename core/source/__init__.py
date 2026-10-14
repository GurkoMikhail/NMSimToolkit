"""
Подпакет источников излучения и компилятора источников расчетного ядра.
"""

from core.source.source_compiler import SourceCompiler
from core.source.sources import PointSource, Source

__all__ = [
    "Source",
    "PointSource",
    "SourceCompiler",
]


