"""FastWilcox - Fast Wilcoxon rank-sum test for single-cell genomics."""

from .models import WilcoxResult
from .wilcox import wilcoxauc

__version__ = "0.1.0"
__all__ = ["wilcoxauc", "WilcoxResult"]
