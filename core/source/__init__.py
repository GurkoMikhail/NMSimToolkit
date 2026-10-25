"""
Подпакет источников излучения, компилятора источников и Numba-ядер инжекции первичных частиц.
"""

from core.source.alias_tables import build_alias_tables
from core.source.source_buffer import SourceBuffer
from core.source.source_compiler import SourceCompiler
from core.source.sources import PointSource, Source

__all__ = [
    "Source",
    "PointSource",
    "SourceBuffer",
    "SourceCompiler",
    "build_alias_tables",
]

