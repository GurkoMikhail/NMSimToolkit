"""
Подпакет источников излучения и компилятора источников расчетного ядра.
"""

from core.source.sources import Source, PointSource
from core.source.source_compiler import SourceCompiler

__all__ = [
    'Source',
    'PointSource',
    'SourceCompiler',
]
