# gui/notation/__init__.py
"""
Notation module for peak processing and database searches.
"""
from .peak_processing import PeakProcessor
try:
    from .bmrb_search import BMRBSearcher
except ImportError:
    # Online BMRB search is optional; local annotation must still start.
    BMRBSearcher = None
from .noise_estimation import NoiseEstimator
from .results_display import ResultsDisplay

__all__ = [
    'PeakProcessor',
    'BMRBSearcher',
    'NoiseEstimator',
    'ResultsDisplay'
]