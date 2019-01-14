"""
Functions for computing binned responses.
"""

from itertools import product

import numpy as np

from roto import circstats


DEFAULT_BINS = 15
INFO_BINS = DEFAULT_BINS

DEFAULT_DIR_BINS = 24
INFO_DIR_BINS = DEFAULT_DIR_BINS

TIME_BINS = 25
TIME_MIN = -4.0
TIME_MAX = 0.5


_norm = lambda H: H / H.sum()

def skaggs(spkmap, occmap, valid):
    """Compute the Skaggs spatial information rate for a spikemap.

    Arguments:
    spkmap -- binned spike-count map
    occmap -- binned occupancy map
    valid -- binned boolean map or index array of valid pixels

    Returns:
    float, spike information rate (bits/spike)
    """
    spk = spkmap[valid]
    occ = occmap[valid]

    if occ.sum() == 0.0:
        return 0.0

    F = spk.sum() / occ.sum()
    if F == 0.0:
        return 0.0

    p = occ / occ.sum()
    f = spk / occ
    nz = np.nonzero(f)

    I = np.sum(p[nz] * f[nz] * np.log2(f[nz] / F)) / F

    return I

def jointtiming(ts, bins=TIME_BINS, tlim=(TIME_MIN, TIME_MAX)):
    """Construct a joint timing diff-diff probability distribution."""
    tmin, tmax = tlim

    te = np.linspace(tmin, tmax, bins + 1)
    edges = [te, te]

    dt = np.log(np.diff(t0))
    dt1 = dt[:-1]
    dt2 = dt[1:]

    H, _, _ = np.histogram2d(dt1, dt2, bins=edges)
    Ht = H.sum()
    if Ht == 0:
        return H
    P = H / Ht
    return P

#
# Below could be used to build information functions
#

def dirinfo(d, Dp):
    """Compute the spike directional information rate."""
    smap = dirspikemap(d, bins=INFO_DIR_BINS)
    omap = diroccmap(Dp, bins=INFO_DIR_BINS)[0]
    valid = omap > 0
    return skaggs(smap, omap, valid)

def dirmap(d, Dp, bins=None, freq=30.0):
    """Construct a 1D movement-direction firing-rate map.

    Arguments:
    d -- spike direction array
    Dp -- movement-direction occupancy array

    Keyword arguments:
    bins -- int or 2-tuple, number of directional bins
    freq -- sampling frequency of the occupancy data

    Returns:
    1D `bins`-shaped array of firing rates, array of angles
    """
    smap = dirspikemap(d, bins=bins)
    omap, angles = diroccmap(Dp, bins=bins, freq=freq)
    valid = omap > 0

    H = np.zeros_like(omap)
    H[valid] = smap[valid] / omap[valid]

    return H, angles

def dirspikemap(ds, bins=None):
    """Directional spike-count map."""
    bins = bins or DEFAULT_DIR_BINS
    return np.histogram(ds, bins=bins, range=MDIR_RANGE)[0]

def diroccmap(d, bins=None, freq=30.0):
    """Directional occupancy map in seconds."""
    bins = bins or DEFAULT_DIR_BINS
    H_occ, edges = np.histogram(d, bins=bins, range=MDIR_RANGE)
    H = H_occ / freq
    centers = (edges[:-1] + edges[1:]) / 2
    return H, centers
