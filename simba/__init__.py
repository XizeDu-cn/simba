"""SIngle-cell eMBedding Along with features."""

import sys

from . import datasets
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from ._settings import settings
from ._version import __version__
from .pipeline import ScATACConfig, run_scatac_pipeline
from .readwrite import *  # noqa: F401,F403

# needed when building docs (borrowed from scanpy)
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl"]})

__all__ = [
    "settings",
    "pp",
    "tl",
    "pl",
    "datasets",
    "ScATACConfig",
    "run_scatac_pipeline",
    "__version__",
]
