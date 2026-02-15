from .koopman import Koopman
from .koopman_continuous import KoopmanContinuous

# Import submodules so they are accessible as attributes
from . import common
from . import differentiation
from . import observables
from . import regression
from . import analytics

__all__ = [
    "Koopman",
    "KoopmanContinuous",
    "common",
    "differentiation",
    "observables",
    "regression",
    "analytics",
]
