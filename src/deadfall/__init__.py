"""deadfall - backdoor detection, trigger reconstruction and removal for tree-ensemble models."""
from .ir import Ensemble
from .loaders import UnsupportedModel, load_model
from .report import ScanReport, scan

__version__ = "0.1.0"
__all__ = ["Ensemble", "ScanReport", "UnsupportedModel", "load_model", "scan", "__version__"]
