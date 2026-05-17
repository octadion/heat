"""
Methods registry. Same as v1.1 — copied for completeness so users can drop
this whole patch folder into their codebase.
"""

from .base import AdaptMethod
from .source import Source
from .bn_adapt import BNAdapt
from .tent import Tent
from .tea import TEA
from .heat import HEAT
from .epotta import EPOTTA
from .retta import ReTTA

__all__ = [
    "AdaptMethod", "Source", "BNAdapt", "Tent", "TEA", "HEAT",
    "EPOTTA", "ReTTA",
]
