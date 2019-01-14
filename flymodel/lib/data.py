"""
Handle HDF5 data paths and object-oriented data access.
"""

import os
import re

import numpy as np
import tables as tb
import pandas as pd
from scipy.interpolate import interp1d

from roto.decorators import lazyprop
from pouty import log

from .. import store


recordings_description = {
    'id': tb.UInt16Col(pos=0),
    'area': tb.StringCol(itemsize=6, pos=1),
    'label': tb.StringCol(itemsize=10, pos=2),
    'timecode': tb.StringCol(itemsize=8, pos=3),
    'phase': tb.StringCol(itemsize=6, pos=4),
    'file': tb.StringCol(itemsize=64, pos=5),
    'path': tb.StringCol(itemsize=64, pos=6)
    }


def recording_path(info):
    """Construct a recording group path from a dict of values."""
    return "/data/{phase}/cell_{label}".format(**info)

def dataframe_from_table(where, name=None, h5file=None, **from_recs_kwds):
    """Return a pandas dataframe for the specified table.

    Arguments:
    where -- Table object to be directly converted to a dataframe
    where/name -- group/path for table with optional name
    h5file -- HDF file [default: project store]

    Remaining keyword arguments are passed to `pandas.DataFrame.from_records`.
    """
    if type(where) is tb.Table:
        table = where
    else:
        f = store.get() if h5file is None else h5file
        table = f.get_node(where, name=name, classname='Table')

    df = pd.DataFrame.from_records(table.read(), **from_recs_kwds)

    # String columns should be decoded from bytes arrays
    for colname, coltype in table.coltypes.items():
        if coltype == 'string':
            df[colname] = df[colname].apply(lambda x: x.decode())

    return df

def session_record(id_label):
    """Recording data dictionary for id or label."""
    recdf = dataframe_from_table('/recordings', index='id')

    if type(id_label) is str:
        try:
            recid = recdf.loc[recdf.label == id_label].index[0]
        except IndexError:
            pass
        else:
            return dict(recdf.loc[recid])

    try:
        recid = int(id_label)
    except ValueError:
        pass
    else:
        return dict(recdf.loc[recid])

    raise ValueError('Invalid id or label: {}'.format(id_label))


class AbstractDataSource(object):

    """
    Abstract base class for loading data traces from a session subgroup.

    Subclasses must set a class attribute called `_subgroup` with the name
    of the group (child of session group) holding the data array nodes.

    Data arrays can have an additional class attribute `<name>_attrs` set
    to a dict of arguments that are passed to `interp1d`.

    For circular data arrays, set boolean key `circular` and, optionally,
    `radians` (default True) and `zero_centered` (default True) to control
    the circular wrapping used in circular-linear interpolation.

    Methods:
    _read_array -- subclasses should use this to read array nodes
    F -- clients should use this to interpolate time-series values
    """

    _subgroup = ''

    def __init__(self, id_or_label):
        """Set up group paths pointing to the recording data.

        Arguments:
        id_or_label -- recording session index or cell label
        """
        rec = session_record(id_or_label)
        where = rec['path']

        f = store.get()
        session = f.get_node(where)
        group = f.get_node(where, name=self.__class__._subgroup)

        self.session_path = str(session._v_pathname)
        self.path = str(group._v_pathname)

        self._interpolators = {}

    def _read_array(self, name):
        """Subclass property attributes should call this to read data."""
        f = store.get()
        try:
            arr = f.get_node(self.path, name)
        except tb.NoSuchNodeError:
            log('array \'{}\' does not exist at {}', name, self.path,
                    error=True, prefix='DataSource')
            return np.array([])
        return arr.read()

    def F(self, which, t_i):
        """Interpolate stored data traces for given time points."""
        try:
            return self._interpolators[which](t_i)
        except KeyError:
            pass

        if not hasattr(self, which):
            raise ValueError("no attribute named %s" % which)

        trace = getattr(self, which)
        if trace.shape[-1] != self.t.size:
            raise ValueError("size mismatch for %s along axis" % which)

        attrs = getattr(self, '%s_attrs' % which, {})
        circular = attrs.get('circular', False)
        bad = attrs.get('bad_value', None)

        ikw = dict(copy=False, bounds_error=False, fill_value=0.0)
        ikw.update({k:attrs[k] for k in attrs
            if k not in ('circular', 'bad_value')})

        intp = circular and circinterp1d or interp1d
        valid = slice(None) if bad is None else (trace != bad)

        self._interpolators[which] = f = \
                intp(self.t[valid], trace[valid], **ikw)

        return f(t_i)


class circinterp1d(object):

    """
    Wrapper for interp1d for circular-linear interpolation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize as `scipy.interpolate.interp1d`."""
        t, angle = args
        self._rads = kwargs.pop('radians', True)
        self._zc = kwargs.pop('zero_centered', True)
        _bad = kwargs.pop('bad_value', None)
        if _bad is not None:
            valid = (angle != _bad)
            t, angle = t[valid], angle[valid]
        self._f = interp1d(t, np.unwrap(angle), **kwargs)

    def __call__(self, t):
        """Perform circular-linear interpolation for the given time points."""
        twopi = self._rads and 2 * np.pi or 360.0
        wrapped = self._f(t) % twopi
        if self._zc:
            if np.iterable(wrapped):
                wrapped[wrapped > 0.5 * twopi] -= twopi
            elif wrapped > 0.5 * twopi:
                wrapped -= twopi
        return wrapped
