"""
Flymodel :: Flymodel
Author :: Joseph Monaco (jmonaco@jhu.edu)
Created :: 2019-01-14
"""

__version__ = '0.0.1'

import os

from tenko.store import DataStore as _Store
from pouty.console import ConsolePrinter as _Printer


# Set up package-wide project paths and metadata

REPO_ROOT = os.path.split(__file__)[0]
PROJECT_TAG = 'flymodel'
PROJECT_ROOT = os.path.join(os.getenv('HOME'), 'projects', PROJECT_TAG)
RES_DIR = os.path.join(PROJECT_ROOT, 'results')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_ROOT = '/Users/joe/data/wulab/flysleep'


# Create a package-wide data store instance for the project data file

store = _Store(name=PROJECT_TAG,
               where=DATA_ROOT,
               logfunc=_Printer(prefix='FlyModel',
                                prefix_color='cyan',
                                message_color='lightgray'))
