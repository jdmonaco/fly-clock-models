"""
Collection of all modeling data plots for paper figures.
"""

import os
from os.path import join as opj

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from roto.dicts import merge_two_dicts

from .sim.diurnal import ClockNeuronModel

from . import RES_DIR as ResultsPath
from .context import FlyAnalysis


# Paths to all analyses used for plotting figures
ClockModelPath = opj(ResultsPath, '2017-12-12+clock-neuron-model+30s')

# Matplotlib settings for publication figures
axes_lw = 0.8
rc_paper = {
        'savefig.format': 'pdf',
        'savefig.frameon': False,
        'savefig.transparent': True,
        'font.size': 9.0,
        'font.weight': 'normal',
        'font.sans-serif': ['Helvetica Neue LT Std', 'Lucida Grande', 'Geneva',
                            'Verdana', 'Bitstream Vera Sans', 'sans-serif'],
        'pdf.fonttype': 'truetype',
        'mathtext.default': 'rm',
        'figure.titlesize': 'large',
        'figure.titleweight': 'light',
        'figure.subplot.wspace': 0.23,
        'figure.subplot.hspace': 0.23,
        'axes.titlesize': 8.0,
        'axes.titleweight': 'light',
        'axes.labelsize': 8.0,
        'axes.linewidth': axes_lw,
        'axes.labelpad': 1.6,
        'lines.solid_capstyle': 'round',
        'lines.linewidth': 1.2,
        'lines.markeredgewidth': 0.0,
        'lines.markersize': 5.6,
        'patch.linewidth': 0.24,
        'xtick.major.width': axes_lw,
        'ytick.major.width': axes_lw,
        'xtick.minor.width': axes_lw/2,
        'ytick.minor.width': axes_lw/2,
        'xtick.major.pad': 1.3,
        'ytick.major.pad': 1.3,
        'xtick.labelsize': 7.2,
        'ytick.labelsize': 7.2,
        'legend.fontsize': 8.0,
        'legend.labelspacing': 0.25
    }
rc_png = {
        'savefig.format': 'png',
        'savefig.frameon': True,
        'savefig.transparent': False,
        'savefig.dpi': 600
    }

mpl.rc_file_defaults()
mpl.rcParams.update(rc_paper)

# Seaborn settings
sns.set_palette('colorblind', color_codes=True)

# Color constants
nonsigcolor = (0.88470588, 0.55803921, 0.3)  # lightened colorblind 'r'
sigcolor = (0.3, 0.61294117, 0.78862745)  # lightened colorblind 'b'
phcolor = (0.3, 0.73372549, 0.61568627)  # lightened colorblind 'g'
poscolor = 'sandybrown'
negcolor = 'slateblue'
noncolor = 'dimgray'

# Savefig parameters
savekw = dict(tight_padding=0.01, closeafter=True)

# Standard reference figure dimensions
col_width = 3.4  # inches
full_width = 7.2
col_height = 8.2
half_height = 0.5*col_height
figsize_hh = (col_width, half_height)
figsize_hf = (col_width, col_height)
figsize_fh = (full_width, half_height)
figsize_ff = (full_width, col_height)

# Common analysis parameters
IBI_MIN = 0.025     # seconds, inter-burst interval threshold


class PaperFigures(FlyAnalysis):

    def run_all_figures(self):
        """Script method to run all figure group methods."""
        self.clock_neuron_models()

    def clock_neuron_models(self):
        """Spike shape and firing pattern data panels for DN1p models."""
        plt.ioff()
        path = self.mkdir('00-clock-neuron-models')
        clock = ClockNeuronModel.load(ClockModelPath)
        clock.set_figfmt('mpl')

        mpl.rcParams['axes.labelpad'] = 1.7
        mpl.rcParams['xtick.major.pad'] = 1.7
        mpl.rcParams['ytick.major.pad'] = 1.7
        mpl.rcParams['figure.subplot.wspace'] = 0.29
        mpl.rcParams['legend.frameon'] = False

        # Main clock neuron models figure
        clock.plot_main_figure(
                h=0.04,
                pad=0.1,
                nspiketraces=200,
                trace_interval=(26,32),
                tracescale=1.0,
                nrasters=6,
                rasterdur=10.0,
                rasterscale=5.0,
                isilim=(0.5,4),
                isibins=32,
                notebook=False,
                figsize=figsize_fh)
        clock.savefig(basepath=opj(path, 'model'), **savekw)

        # Restore rc settings
        mpl.rcParams.update(rc_paper)
