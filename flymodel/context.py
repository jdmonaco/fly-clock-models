"""
Analysis/simulation context classes for flymodel project.
"""

import os

from tenko.context import step
from tenko.factory import ContextFactory

from . import DATA_ROOT, REPO_ROOT, RES_DIR, PROJECT_ROOT


FlyAnalysis = ContextFactory.analysis_class("FlyAnalysis",
    os.path.join(PROJECT_ROOT, 'ana'), REPO_ROOT, RES_DIR, logcolor='purple')

FlySimulation = ContextFactory.analysis_class("FlySimulation",
    os.path.join(PROJECT_ROOT, 'sim'), REPO_ROOT, RES_DIR, logcolor='yellow')
