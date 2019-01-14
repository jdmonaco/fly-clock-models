"""
Functions for computing binned responses.
"""

import numpy as np


DELTA_BINS = 25
DELTA_LIM = (-2.5, 1.8)
DELTA_MIN = DELTA_LIM[0]
DELTA_MAX = DELTA_LIM[1]


def condsamples(tslist, bins=DELTA_BINS, dlim=DELTA_LIM):
    """Collate a list of spike trains into normalized conditional samples."""
    dmin, dmax = dlim

    deltas = None
    for ts in tslist:
        nspikes = ts.size
        t0 = ts - ts[0]
        tnorm = t0 * (nspikes / t0.max())

        dt = np.log(np.diff(tnorm))
        dt_pairs = np.c_[dt[:-1], dt[1:]]

        if deltas is None:
            deltas = dt_pairs
        else:
            deltas = np.concatenate((deltas, dt_pairs))

    # Convert first column (dt1) to a bin index for collation
    dt1 = deltas[:,0]
    ix = np.floor(bins * (dt1 - dmin) / (dmax - dmin)).astype('i')

    # Clip samples outside of the valid range
    valid = np.all((deltas > dmin) & (deltas < dmax), axis=1)
    ix = ix[valid]
    deltas = deltas[valid]

    # Return bin index and normalized-dt pairs array
    return ix, deltas

def conddelta(ts, bins=DELTA_BINS, dlim=DELTA_LIM):
    """Construct a normalized conditional timing distribution."""
    dmin, dmax = dlim

    de = np.linspace(dmin, dmax, bins + 1)
    edges = [de, de]

    nspikes = ts.size
    t0 = ts - ts[0]
    tnorm = t0 * (nspikes / t0.max())

    dt = np.log(np.diff(tnorm))
    dt1 = dt[:-1]
    dt2 = dt[1:]

    H, _, _ = np.histogram2d(dt1, dt2, bins=edges)
    P = H / H.sum(axis=1).reshape((-1,1))
    P[~np.isfinite(P)] = 0.0  # zero-out columns with zero sums

    return P
