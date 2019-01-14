"""
Functions for computing characteristics of spike trains.
"""

from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy import trapz

import pouty as pty
from roto.arrays import maxima, find_groups


BURST_ISI = 0.012     # s
SPIKE_HWIN = 0.004    # s
SPIKE_DV = 4.0        # mV
SPIKE_WAIT = 0.008    # s
SPIKE_CUT = 0.9       # fraction
KERNEL_WIDTH = 5.0    # s
SAMPLE_RATE = 1000.0  # samples/s
LOWPASS_TAU = 30.0


def find_bursts(ts, min_len=2, isi_thresh=BURST_ISI):
    """Find bursts within a spike train.

    Arguments:
    ts -- (n_spikes,) array of spike times in seconds
    min_len -- int >= 2, minimum spike length of bursts to find
    isi_thresh -- maximum ISI cutoff for intraburst spikes

    Returns:
    list of spike-time array slices with burst spikes
    """
    assert min_len > 1, 'min_len must be >=2'
    burst_spikes = np.diff(ts) <= isi_thresh
    groups = find_groups(burst_spikes, min_size=min_len - 1)

    bursts = []
    for g in groups:
        bursts.append(ts[slice(g[0], g[1] + 1)])

    return bursts

def burst_fraction(ts, min_len=2, isi_thresh=BURST_ISI):
    """Calculate fraction of spikes participating in bursts.

    Same arguments as `find_bursts`. Returns float value on [0,1].
    """
    assert min_len > 1, 'min_len must be >=2'
    burst_spikes = np.diff(ts) <= isi_thresh
    groups = find_groups(burst_spikes, min_size=min_len - 1)

    bursts = 0
    for g in groups:
        bursts += g[1] - g[0] + 1

    return bursts / ts.size

def find_spikes(t, v, half_window=SPIKE_HWIN, dv_thresh=SPIKE_DV,
    wait=SPIKE_WAIT, cut_pct=SPIKE_CUT, return_voltage=False):
    """Detect spikes in a patch trace voltage signal."""
    vsort = np.sort(v)
    cut = vsort[int(v.size*cut_pct)]

    ix = maxima(v)
    valid = ix[v[ix] > cut]
    tp = t[valid]
    vp = v[valid]
    N = len(valid)

    last = -1
    spikes = []
    for i in range(N):
        if tp[i] - last <= wait:
            continue

        lwin = (t >= tp[i] - half_window) & (t < tp[i])
        uwin = (t > tp[i]) & (t <= tp[i] + half_window)
        lflag = vp[i] - v[lwin].min() >= dv_thresh
        uflag = vp[i] - v[uwin].min() >= dv_thresh
        if lflag & uflag:
            last = tp[i]
            if return_voltage:
                spikes.append((tp[i], vp[i]))
            else:
                spikes.append(tp[i])

    return np.array(spikes)

def spike_traces(ts, ttrace, vtrace, half_window=SPIKE_WAIT):
    """Spike-triggered slices of a voltage trace."""
    delta = None
    traces = None
    start = ttrace.min()
    end = ttrace.max()
    dt = np.median(np.diff(ttrace[:1000]))
    hbins = int(half_window/dt)
    wbins = 2 * hbins + 1

    for t in ts:
        if t <= start + half_window or t >= end - half_window:
            continue

        tix = np.nonzero(t == ttrace)[0][0]
        ix = slice(tix - hbins, tix - hbins + wbins)
        tr = vtrace[ix]
        if traces is None:
            traces = tr.values.reshape(-1,1)  # initiate column vector
            delta = ttrace[ix] - t
        else:
            traces = np.concatenate((traces, tr[:,np.newaxis]), axis=1)

    return delta, traces

def lowpass_filter(v, tau=LOWPASS_TAU):
    """Run a low-pass filter on the voltage trace to extract slow component."""
    M = int(tau*SAMPLE_RATE)
    vpad = np.r_[np.zeros(M) + v[0], v, np.zeros(M) + v[-1]]

    box = sig.boxcar(M) / M
    vbox = sig.convolve(vpad, box, mode='same', method='fft')

    gsn = sig.gaussian(M, (tau / 6.0) * SAMPLE_RATE)
    gsn = gsn / trapz(gsn)
    vgsn = sig.convolve(vbox, gsn, mode='same')

    vlowp = vgsn[M:-M]
    return vlowp


class FiringRateEstimate(object):

    """
    Compute a binless, kernel-based estimate of instantaneous firing rate.
    """

    def __init__(self, spikes, duration, width=KERNEL_WIDTH, kernel='gaussian',
        Fs_norm=60.0):
        self.out = pty.ConsolePrinter(prefix=self.__class__.__name__)
        if spikes.ndim != 2:
            spikes = np.atleast_2d(spikes).T

        self.spikes = spikes
        self.avg_rate = spikes.size / duration
        self.sigma = sigma = width / np.sqrt(12)
        self.model = neighbors.KernelDensity(bandwidth=sigma, kernel=kernel,
            rtol=1e-3)
        self.model.fit(spikes)
        if Fs_norm == 0.0:
            self.norm = 1.0
        else:
            self._normalize(Fnorm=Fs_norm)

    def _normalize(self, Fnorm=60.0):
        self.out('Normalizing firing rate estimate ({} spikes)', self.spikes.size)
        smin, smax = self.spikes[0], self.spikes[-1]
        t0, t1 = smin - 2 * self.sigma, smax + 2 * self.sigma
        tp = np.linspace(t0, t1, int(Fnorm * (t1 - t0)))
        tp = np.atleast_2d(tp).T
        pf = np.exp(self.model.score_samples(tp)).squeeze()
        self.norm = self.avg_rate / pf.mean()

    def evaluate(self, tp):
        """Evaluate the firing-rate estimate at an array of time points."""
        if tp.ndim != 2:
            tp = np.atleast_2d(tp).T

        logp = self.model.score_samples(tp).squeeze()
        return self.norm * np.exp(logp)

    __call__ = evaluate
