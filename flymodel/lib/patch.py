"""
Data access object for voltage trace data from whole-cell patch.
"""

from roto.decorators import lazyprop

from .data import AbstractDataSource


class PatchTrace(AbstractDataSource):

    """
    Load continuous data traces for membrane voltage recordings.
    """

    _subgroup = ''

    @lazyprop
    def t(self):
        ts = self._read_array('T')
        return ts / 1000.0  # convert to seconds

    @lazyprop
    def v(self):
        return self._read_array('V')

    @lazyprop
    def v_slow(self):
        return self._read_array('V_lowpass')

    @lazyprop
    def r(self):
        return self._read_array('rate_slow')


class PatchSpikes(AbstractDataSource):

    """
    Load detected spike-timing data for patch recordings.
    """

    _subgroup = 'spikes'

    @lazyprop
    def t(self):
        ts = self._read_array('T')
        return ts

    @lazyprop
    def v(self):
        return self._read_array('V')
