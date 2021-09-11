"""Type Aliases used in the autoRS package."""

# Standard library imports
from __future__ import annotations
from typing import Tuple, List, Union

# Third party imports
import numpy as np

# Type Aliases. Docstrings provided for Sphinx autodoc.
numeric = Union[int, float]

array_like_1d = Union[List[numeric], Tuple[numeric], np.ndarray]
"""Package specific 1D type-alias for inputs/outputs of RS functions."""

array_like_2d = Union[List[array_like_1d], Tuple[array_like_1d], np.ndarray]
"""Package specific 2D type-alias for inputs/outputs of RS functions."""
