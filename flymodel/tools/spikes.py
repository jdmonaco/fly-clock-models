"""
Functions for computing characteristics of spike trains.
"""

from ipyparallel import interactive
import matplotlib.pyplot as plt
import numpy as np

import tenko.parallel
import pouty as pty


def acorr(t, **kwargs):
    """Compute spike train autocorrelograms."""
    return xcorr(t, t, **kwargs)

def xcorr(a, b, maxlag=1.0, bins=128, side=None, parallel=False):
    """Compute the spike train correlogram of two spike train arrays

    Arguments:
    a, b -- compute correlations of spike train b relative to spike train a

    Keyword arguments:
    maxlag -- range of lag times (+/-) to be returned
    bins -- number of discrete lag bins to use for the histogram
    side -- None|'left'|'negative'|'right'|'positive', restrict lag range
    parallel -- use ipyparallel implementation

    Returns:
    (counts, bin_centers) tuple of float arrays
    """
    lagmin, lagmax = -maxlag, maxlag
    if side is None:
        if bins % 2 == 0:  # zero-center range should
            bins += 1      # include a zero-center bin
            pty.debug('xcorr: zero-center correlation, so '
                      'changing bins from {} to {}'.format(bins-1, bins))
    elif side in ('left', 'negative'):
        lagmax = 0.0
    elif side in ('right', 'positive'):
        lagmin = 0.0
    else:
        pty.debug('xcorr: ignoring unknown side value ({})'.format(repr(side)))
    return _xcorr(a, b, lagmin, lagmax, bins, parallel)

def _xcorr(a, b, lagmin, lagmax, bins, parallel):
    edges = np.linspace(lagmin, lagmax, bins + 1)

    if parallel:
        rc = tenko.parallel.client()
        dview = rc[:]
        dview.block = True
        dview.execute('import numpy as np')
        dview.scatter('_xcorr_spiketrain', a)
        dview['_xcorr_reference'] = b
        dview['_xcorr_edges'] = edges
        dview['_xcorr_kernel'] = _xcorr_kernel
        ar = dview.apply_async(_xcorr_parallel, lagmin, lagmax, bins)
        H = np.array(ar.get()).sum(axis=0)
    else:
        H = _xcorr_kernel(a, b, lagmin, lagmax, bins, edges)

    centers = (edges[:-1] + edges[1:]) / 2
    return H, centers

@interactive
def _xcorr_parallel(lagmin, lagmax, bins):
    a, b, edges = _xcorr_spiketrain, _xcorr_reference, _xcorr_edges
    return _xcorr_kernel(a, b, lagmin, lagmax, bins, edges)

def _xcorr_kernel(a, b, lagmin, lagmax, bins, edges):
    nb = b.size
    start = end = 0
    H = np.zeros(bins)
    for t in a:
        while start < nb and b[start] < t + lagmin:
            start += 1
        if start == nb:
            break
        while end < nb and b[end] <= t + lagmax:
            end += 1
        H += np.histogram(b[start:end] - t, bins=edges)[0]
    return H

def maxlag_from_lags(lags):
    """From an array of lag bin centers, back-calculate the maxlag."""
    lags = np.asarray(lags)
    bins = lags.size
    maxlag = lags.max() + (lags.ptp() / (bins - 1) / 2)
    return maxlag

def corrplot(data, ax=None, style='verts', zero_line=True, norm=False, **fmt):
    """Plot spike train correlogram to the specified axes.

    Remaining keyword arguments are passed to `Axes.plot(...)` (lines, steps)
    or `Axes.vlines(...)` (verts).

    Arguments:
    data -- spike train array (acorr) or tuple of two arrays (xcorr)

    Keyword arguments:
    ax -- axes object to draw autocorrelogram into
    style -- plot style: can be 'verts' (vlines), 'steps', or 'lines'
    zero_line -- draw a vertical dotted line at zero lag for reference
    norm -- normalize so that the peak correlation is 1

    Returns:
    plot handle
    """
    ax = ax is None and plt.gca() or ax
    C, lags = data
    if norm:
        C = C.astype(float) / C.max()

    if style in ('lines', 'steps'):
        fmt.update(lw=fmt.get('lw', 2))
        if style == 'steps':
            fmt.update(drawstyle='steps-mid')
        h = ax.plot(lags, C, **fmt)
        h = h[0]
    elif style == 'verts':
        fmt.update(colors=fmt.get('colors', 'k'), lw=fmt.get('lw',2))
        h = ax.vlines(lags, [0], C, **fmt)

    ax.axhline(color='k')
    if zero_line:
        ax.axvline(color='k', ls=':')
    ax.set_xlim(lags.min(), lags.max())

    plt.draw_if_interactive()
    return h
