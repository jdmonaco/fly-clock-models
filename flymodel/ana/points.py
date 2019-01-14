"""
Model timing structure of day/night cells as point samples.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as sit
import scipy.stats as st

import roto.axis as ra

from .. import store
from ..context import FlyAnalysis, step
from ..lib import activity, data, patch, estimation
from ..tools import binned


DURATION = 60.0         # s
DEFAULT_BINS = 22
DEFAULT_DMIN = -2.225
DEFAULT_DMAX = 1.4


class PointProcessModel(FlyAnalysis):

    """
    A sampling point process model for synthetic spike trains.
    """

    def cleanup(self):
        store.close()

    @step
    def setup(self, phase='day', bins=DEFAULT_BINS, dmin=DEFAULT_DMIN,
        dmax=DEFAULT_DMAX):
        """Setup model fitting for day or night phase."""
        self['phase'] = phase
        self['bins'] = bins
        self['dmin'] = dmin
        self['dmax'] = dmax

        # Store a list of cell labels
        rdf = data.dataframe_from_table('/recordings', index='id')
        self['labels'] = list(rdf.loc[rdf.phase == phase, 'label'])

    def script(self, figfmt='mpl'):
        """Run all main methods and save figures."""
        self.set_figfmt(figfmt)
        plt.ioff()

        self.grand_conditional()
        self.store_samples()
        self.estimate_conditional()
        self.interpolate_conditional()
        self.marginalize_conditional()
        self.joint_comparison(share_scale=False)
        self.validate_synthesis()

        self.save_figures()
        plt.ion()
        plt.show()

    @step
    def grand_conditional(self, figsize=(5,4)):
        """Grand average of normalized conditional timing distributions."""
        dists = []
        for label in self.c.labels:
            pt = patch.PatchSpikes(label)
            Pdt = estimation.conddelta(pt.t, bins=self.c.bins,
                    dlim=(self.c.dmin, self.c.dmax))
            dists.append(Pdt)

        # Sum, normalize, and validate the grand conditional distribution
        avg = np.sum(dists, axis=0)
        P = avg / avg.sum(axis=1).reshape((-1,1))
        P[~np.isfinite(P)] = 0.0

        self.save_array('P_grand', P,
            title='Grand Normalized Conditional Distribution')

        f = self.figure('grand-distro', clear=True,
                figsize=figsize, title='Grand Conditional Normalized Timing: '
                    f'{self.c.phase.title()} Cells')
        ax = f.add_subplot(111)

        extent = [self.c.dmin, self.c.dmax] * 2
        ax.imshow(P.T, cmap='cubehelix', vmin=0, extent=extent, aspect='equal')

        ra.despine(ax)
        ax.set(ylabel=r'log $\Delta{t_2^*}$',
               xlabel=r'log $\Delta{t_1^*}$')

    @step
    def store_samples(self):
        """Store normalized timing samples for parameter estimation."""
        spike_trains = [patch.PatchSpikes(label).t for label in self.c.labels]
        ix, deltas = estimation.condsamples(spike_trains, bins=self.c.bins,
                dlim=(self.c.dmin, self.c.dmax))

        N = len(deltas)
        counts = [np.sum(ix == i) for i in range(self.c.bins)]
        self.out('Total deltas: {}', N)
        self.out('Total samples: {}', np.sum(counts))
        self.out('Sample bin counts: {}', counts)

        # Convert sample object array to dataframe for storage
        sdf = pd.DataFrame(data={'bin': ix, 'dt1': deltas[:,0],
            'dt2': deltas[:,1]})
        self.save_dataframe('estimation/samples', sdf,
                title='Conditional Binned Normalized Samples')

    @step
    def estimate_conditional(self, nevals=128, figsize=(9,4.5)):
        """Estimate binned conditional distributions for timing."""
        sdf = self.read_dataframe('estimation/samples')

        x = np.linspace(self.c.dmin, self.c.dmax, nevals)
        params = np.empty((self.c.bins, 2), 'd')
        pdfs = np.empty((self.c.bins, x.size), 'd')
        histos = np.empty((self.c.bins, self.c.bins), 'd')

        edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
        binwidth = (self.c.dmax - self.c.dmin) / self.c.bins

        loc, scale = self.c.dmin, self.c.dmax - self.c.dmin
        self['loc'] = loc
        self['scale'] = scale

        for i in range(self.c.bins):
            data = sdf.loc[sdf.bin == i, 'dt2']
            a, b, _, _ = st.beta.fit(data, floc=loc, fscale=scale)

            params[i,:] = a, b
            pdfs[i,:] = st.beta.pdf(x, a, b, loc=loc, scale=scale)
            histos[i,:], _ = np.histogram(data, bins=edges)

        pardf = pd.DataFrame(data=params, columns=['a', 'b'])
        self.save_dataframe('conditional/params', pardf,
                title='Beta Parameters for Binned Conditional Distributions')

        f = self.figure('conditional-distro', clear=True, figsize=figsize,
                title='Binned Conditional Distribution Estimates: '
                    f'{self.c.phase.title()} Cells')
        ax = f.add_subplot(111)

        cols = mpl.cm.hsl(np.linspace(0,1,self.c.bins))

        for i in range(self.c.bins):
            valid = np.isfinite(pdfs[i])
            pdf = pdfs[i,valid]
            x_pdf = (-1 / pdf.max()) * pdf + 1
            x_pdf = x_pdf * binwidth + edges[i]
            ax.plot(x_pdf, x[valid], c='k', alpha=0.8, zorder=1)

            ax.bar( edges[i+1], binwidth,
                    width=-binwidth * histos[i]/histos[i].max(),
                    bottom=edges[:-1], color=cols[i], edgecolor='k',
                    linewidth=0.25, orientation='horizontal', align='edge')

        ax.set(xlabel=r'Normalized timing bins of $d_1$', ylabel='Normalized '
                      r'timing $d_2$')
        ax.set_xlim(self.c.dmin, self.c.dmax)
        ax.set_ylim(self.c.dmin, self.c.dmax)
        ax.axis('auto')
        ra.despine(ax)

    def _conditional_interpolants(self):
        paramdf = self.read_dataframe('conditional/params')
        a = paramdf.a
        b = paramdf.b

        edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        f_a = sit.interp1d(centers, a, bounds_error=False,
                fill_value=(a.iloc[0], a.iloc[-1]))
        f_b = sit.interp1d(centers, b, bounds_error=False,
                fill_value=(b.iloc[0], b.iloc[-1]))

        return f_a, f_b

    @step
    def interpolate_conditional(self, ninterps=128, figsize=(8,4)):
        """Fit polynomials to binned conditional beta parameters."""
        paramdf = self.read_dataframe('conditional/params')

        edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        f_a, f_b = self._conditional_interpolants()

        f = self.figure('conditional-interp', clear=True, figsize=figsize,
                title='Parameter Interpolation for Conditionals: '
                    f'{self.c.phase.title()} Cells')
        ax = f.add_subplot(111)

        cols = 'slateblue', 'forestgreen'
        x = np.linspace(self.c.dmin, self.c.dmax, ninterps)

        ax.plot(centers, paramdf.a, ls='', marker='x', c=cols[0], label='a')
        ax.plot(x, f_a(x), ls=':', c=cols[0], label=r'F$_a$')

        ax.plot(centers, paramdf.b, ls='', marker='x', c=cols[1], label='b')
        ax.plot(x, f_b(x), ls=':', c=cols[1], label=r'F$_b$')

        ax.set(xlabel='Normalized timing', ylabel=r'$\beta$ parameters')
        ax.legend()
        ra.despine(ax)

    @step
    def marginalize_conditional(self, ninterps=128, figsize=(7,6)):
        """Marginalize model conditional into marginal and joint estimates to
        compare with estimated conditional and data samples.
        """
        x = np.linspace(self.c.dmin, self.c.dmax, ninterps)
        f_a, f_b = self._conditional_interpolants()

        # Generate parametrically interpolated conditional distribution
        cond = np.array([st.beta.pdf(x, f_a(x0), f_b(x0), loc=self.c.loc,
                scale=self.c.scale) for x0 in x])
        cond[~np.isfinite(cond)] = 0.0  # there may be a few infs

        # Construct marginal from original timing samples
        def construct_data_marginal():
            sdf = self.read_dataframe('estimation/samples')
            deltas = np.r_[sdf.dt1, sdf.dt2.iloc[-1]]
            edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2

            H, _ = np.histogram(deltas, bins=edges)
            counts = np.r_[0, H, 0]

            xmarg = np.r_[self.c.dmin, centers, self.c.dmax]
            c_int = sit.interp1d(xmarg, counts)(x)
            marg = c_int / np.trapz(c_int, x)
            return marg
        data_marg = construct_data_marginal()

        # Multiply marginal by conditionals for synthetic joint distribution
        joint = data_marg[:,np.newaxis] * cond
        joint /= np.trapz(np.trapz(joint, x), x)
        self.save_array('P_joint_synth', joint, title='Synthetic Joint '
            'Distribution')

        # Sum and normalize synthetic joint for synthetic marginal distribution
        synth_marg = joint.sum(axis=0)
        synth_marg /= np.trapz(synth_marg, x)
        self['synth_marg_mode'] = x[np.argmax(synth_marg)]
        self.save_array('P_marg_synth', synth_marg, title='Synthetic Marginal '
                    'Distribution')
        self.save_array('x_synth', x, title='X-Values for Synthetic '
            'Distributions')

        #
        # Create a 2x2 figure showing (sample, model) x (conditional, marginal)
        #

        mpl.rcParams['figure.subplot.wspace'] = 0.26
        mpl.rcParams['figure.subplot.bottom'] = 0.1

        f = self.figure('model-data-comparison', clear=True, figsize=figsize,
                title='Sample (top) and Model (bottom) Conditional and '
                    'Marginal Distributions')
        axdc = f.add_subplot(221)  # data conditional
        axm = f.add_subplot(222)   # data and model marginals
        axc = f.add_subplot(223)   # model conditional

        def plot_cond(ax, P, label, show_xlabel=False):
            im = ax.imshow(P.T, vmin=0.0, aspect='equal', cmap='cubehelix',
                    extent=[self.c.dmin, self.c.dmax] * 2)
            ax.axis('auto')
            if show_xlabel:
                axc.set_xlabel(xlabel=r'Normalized timing $d_1$')
            ax.set(ylabel=r'Normalized timing $d_2$')
            ra.quicktitle(ax, rf'{label.title()} P($d_2|d_1$)', size='small',
                    weight='light')
            ra.despine(ax)
        plot_cond(axdc, self.read_array('P_grand'), 'Data')
        plot_cond(axc, cond, 'Model', show_xlabel=True)

        cols = mpl.cm.Dark2(np.linspace(0,1,3))

        def plot_marg(ax, marg, label, col, ls, m):
            ax.plot(x, marg, color=col, ls=ls, alpha=0.9, marker=m,
                    label=label)
        plot_marg(axm, data_marg, 'Data', cols[0], '--', None)
        plot_marg(axm, synth_marg, 'Model', cols[1], '-', None)

        def finish_marg(ax):
            ax.set(xlabel='Normalized timing', ylabel='Pr[timing]')
            ra.quicktitle(ax, 'Marginals', size='small', weight='light')
            ax.set_ylim(bottom=0.0)
            ra.despine(ax)
            ax.legend()
        finish_marg(axm)

        mpl.rc_file_defaults()

    @step
    def joint_comparison(self, share_scale=True, figsize=(8,4)):
        """Grand and synthetic joint normalized timing distributions."""
        P_joint_synth = self.read_array('P_joint_synth')

        def compute_grand_joint():
            edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            sdf = self.read_dataframe('estimation/samples')
            H, _, _ = np.histogram2d(sdf.dt1, sdf.dt2, bins=[edges, edges])
            grand = H / np.trapz(np.trapz(H, centers), centers)
            return grand
        P_joint_grand = compute_grand_joint()

        # Find cross-comparison max if color scale is shared
        Pmax = None
        if share_scale:
            Pmax = max(P_joint_grand.max(), P_joint_synth.max())

        f = self.figure('joint-comparison', clear=True, figsize=figsize,
                title='Comparison of Data (left) and Model (right) '
                    'Joint Timing Distributions')
        axg = f.add_subplot(121)
        axs = f.add_subplot(122)

        extent = [self.c.dmin, self.c.dmax] * 2

        def plot_joint(ax, P, title, show_ylabel=True):
            ax.imshow(P.T, vmin=0.0, vmax=Pmax, aspect='equal',
                    cmap='cubehelix', extent=extent)
            ax.axis('auto')
            ra.quicktitle(ax, title, size='small', weight='light')
            ax.set(xlabel=r'Normalized timing $d_2$')
            if show_ylabel:
                ax.set(ylabel=r'Normalized timing $d_1$')
            ra.despine(ax)

        plot_joint(axg, P_joint_grand, 'Data')
        plot_joint(axs, P_joint_synth, 'Model', show_ylabel=False)

    @step
    def validate_synthesis(self, seed='wake-sleep', burn_samples=200,
        figsize=(7,6)):
        """Sample the synthetic timing model and check fit to data sample."""
        state = np.random.RandomState(seed=sum(map(ord, seed)))
        f_a, f_b = self._conditional_interpolants()

        delta = self.c.synth_marg_mode  # initialize with mode value
        for _ in range(burn_samples):
            delta = st.beta.rvs(f_a(delta), f_b(delta), loc=self.c.loc,
                        scale=self.c.scale, random_state=state)

        nsamples = self.read_dataframe('estimation/samples').shape[0] + 1
        samples = np.empty((nsamples,), 'd')

        for i in range(nsamples):
            delta = st.beta.rvs(f_a(delta), f_b(delta), loc=self.c.loc,
                        scale=self.c.scale, random_state=state)
            samples[i] = delta

        # Compute distributions of the synthetic sample
        dt1 = samples[:-1]
        dt2 = samples[1:]
        edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
        H, _, _ = np.histogram2d(dt1, dt2, bins=edges)
        joint = H / H.sum()
        cond = H / H.sum(axis=1).reshape((-1,1))
        cond[~np.isfinite(cond)] = 0.0  # kill any infs

        # Load theoretical synthetic marginal for comparison
        marg_synth = self.read_array('P_marg_synth')
        x_synth = self.read_array('x_synth')
        ninterps = x_synth.size

        #
        # Create a figure showing sample conditional & joint and sample
        # marginal histogram with expected synthetic marginal
        #

        mpl.rcParams['figure.subplot.wspace'] = 0.26
        mpl.rcParams['figure.subplot.bottom'] = 0.1

        f = self.figure('validate-synthesis', clear=True, figsize=figsize,
                title=f'Synthetic Sample Distributions: N = {nsamples}')
        axc = f.add_subplot(221)  # synthetic conditional
        axm = f.add_subplot(222)  # synthetic samples and expected marginal
        axj = f.add_subplot(223)  # synthetic joint

        extent = [self.c.dmin, self.c.dmax] * 2

        def plot_cond_or_marg(ax, P, title, xlabel=True):
            im = ax.imshow(P.T, vmin=0.0, aspect='equal', cmap='cubehelix',
                    extent=extent)
            ax.axis('auto')
            if xlabel:
                ax.set_xlabel(r'Normalized timing $d_1$')
            ax.set_ylabel(r'Normalized timing $d_2$')
            ra.quicktitle(ax, title, size='small', weight='light')
            ra.despine(ax)
        plot_cond_or_marg(axc, cond, r'Synthetic P($d_2|d_1$)', xlabel=False)
        plot_cond_or_marg(axj, joint, r'Synthetic P($d_1,d_2$)')

        cols = mpl.cm.Dark2(np.linspace(0,1,3))

        def plot_sample_marg(ax, data, x, marg):
            edges = np.linspace(self.c.dmin, self.c.dmax, ninterps + 1)
            H, _, _ = ax.hist(data, bins=edges, normed=True, color='turquoise',
                    alpha=0.8, zorder=-1, label='Counts')
            ax.plot(x, marg, c=cols[1], ls=':', lw=1.5, zorder=1,
                    label='Expected')

            ax.set(xlabel='Normalized timing', ylabel='Pr[timing]')
            ax.set_ylim(bottom=0.0)
            ra.quicktitle(ax, 'Synthetic samples', size='small',
                    weight='light')
            ra.despine(ax)
            ax.legend()
        plot_sample_marg(axm, samples, x_synth, marg_synth)

        mpl.rc_file_defaults()
