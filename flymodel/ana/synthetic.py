"""
Sample the Gaussian mixture models to generated match-rate pairs for delivery.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from roto.paths import uniquify
from roto.images import tiling_dims
import roto.axis as ra

from .mixture import MixtureModel
from ..context import FlyAnalysis, step
from .. import RES_DIR


DAYDIR = os.path.join(RES_DIR, '2017-09-17+mixture-model+final+day')
NIGHTDIR = os.path.join(RES_DIR, '2017-09-17+mixture-model+night')
DURATION = 3600.0
BINWIDTH = 0.010    # 10 ms stimulus dt


class StimulusPatterns(FlyAnalysis):

    """
    Generate a package of stimulus patterns for delivery.
    """

    @step
    def setup(self, daydir=DAYDIR, nightdir=NIGHTDIR):
        self['daydir'] = daydir
        self['nightdir'] = nightdir

    @step
    def run(self, n, duration=DURATION, label=None, seed='stimulus-patterns'):
        """Generate a batch of day and night stimulus trains with matched
        randomly-sampled mean firing rates.

        Arguments:
        n -- number of stimulus train pairs to generate
        duration -- stimulus train duration in seconds
        label -- name for this set of stimulus train (used for subfolder)

        The generated stimulus trains are saved as {0,1}-valued csv files
        in an anaylsis subfolder.
        """
        day_model = MixtureModel.load(self.c.daydir)
        night_model = MixtureModel.load(self.c.nightdir)
        path = self.subfolder(f"{label or 'synthetic'}-trains")
        _, subf = os.path.split(path)
        figpath = self.mkdir(subf, 'figures')

        params = dict(n=n, burn_samples=200, duration=duration, seed=seed,
                ret_trains=True, plot=False)

        rates, day_trains = day_model.generate_spike_trains(**params)
        _, night_trains = night_model.generate_spike_trains(firing_rate=rates,
                **params)

        edges = np.arange(0, duration + BINWIDTH, BINWIDTH)

        def save_train(i, phase, rate, train):
            stem = os.path.join(path, f'{i:03d}-{rate:.2f}-{phase}')
            fn = stem.replace('.', '_') + '.csv'
            H, _ = np.histogram(train, bins=edges)
            Hstr = ','.join(map(lambda c: str(min(1, c)), H)) + ','
            with open(fn, 'wt', encoding='ascii') as fd:
                fd.write(Hstr)
                fd.write('\n')
            self.out(f'Saved: {fn}')
            return H

        for i in range(n):
            Hday = save_train(i, 'day', rates[i], day_trains[i])
            Hnight = save_train(i, 'night', rates[i], night_trains[i])
            self.save_plots(i, figpath, rates[i], Hday, Hnight)

    def save_plots(self, num, figpath, rate, Hday, Hnight, figsize=(8,4.5)):
        """Save figure of wrapped histograms of stimulus train pair."""
        stem = os.path.join(figpath, f'{num:03d}-{rate:.2f}-pair')
        savepath = stem.replace('.', '_') + '.png'

        f = plt.figure(num='stimplots', figsize=figsize)
        plt.suptitle(f'Stimulus Train Pair #{num:03d}: {rate:.2f} pulses/s')
        axd = f.add_subplot(121)
        axn = f.add_subplot(122)

        def plot_histo(ax, H, title):
            r, c = tiling_dims(H.size)
            pad = r * c - H.size
            Hwrap = np.r_[H, np.zeros(pad)].reshape(r, c)
            ax.imshow(Hwrap, cmap='gray_r', origin='upper', aspect='equal')
            ax.set_axis_off()
            ra.quicktitle(ax, title, va='bottom', size='medium', weight='light')

        plot_histo(axd, Hday, "'Day' train")
        plot_histo(axn, Hnight, "'Night' train")

        plt.tight_layout()
        plt.savefig(savepath)
        plt.close('stimplots')
