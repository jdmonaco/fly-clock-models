"""
Mixture model of joint timing distribution for synthesis.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.mixture as gmm
import scipy.stats as st
import scipy.linalg as la
import seaborn as sns
sns.reset_orig()

import roto.axis as ra
import pouty as pty

from .. import store
from ..context import FlyAnalysis, step
from ..lib import data, patch, estimation


DURATION = 60.0         # s
DEFAULT_BINS = 22
DEFAULT_DMIN = -2.225
DEFAULT_DMAX = 1.4


class MixtureModel(FlyAnalysis):

    """
    Gaussian mixture model of spike timing for synthetic spike trains.
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

    def script(self, n_components=5, figfmt='mpl'):
        """Run every step of the analysis and save figures."""
        self.set_figfmt(figfmt)
        plt.ioff()

        self.store_samples()
        self.fit_mixture_model(n_components=n_components)
        self.plot_mixture_model()
        self.evaluate_conditional()
        self.validation_sample()
        self.plot_validation_sample()
        self.fit_rate_distro()
        self.generate_spike_trains()

        self.save_figures()
        plt.ion()
        plt.show()

    @step
    def store_samples(self):
        """Store normalized timing samples for parameter estimation."""
        spike_trains = [patch.PatchSpikes(label).t for label in self.c.labels]
        ix, deltas = estimation.condsamples(spike_trains, bins=self.c.bins,
                dlim=(self.c.dmin, self.c.dmax))

        counts = [np.sum(ix == i) for i in range(self.c.bins)]
        ncounts = int(np.sum(counts))
        self['N_deltas'] = ncounts + 1  # account for dt2 in last pair

        self.out('Total delta pairs: {}', len(deltas))
        self.out('Total delta pairs (in bins): {}', ncounts)
        self.out('Bin counts: {}', counts)

        sdf = pd.DataFrame(data={'bin': ix, 'dt1': deltas[:,0],
            'dt2': deltas[:,1]})
        self.save_dataframe('estimation/samples', sdf,
                title='Conditional Binned Normalized Samples')

    @step
    def fit_mixture_model(self, n_components=5, seed='fit-mixture-model'):
        """Fit the gaussian mixture model to the second-order timing data."""
        state = np.random.RandomState(seed=sum(map(ord, seed)))
        sdf = self.read_dataframe('estimation/samples')
        dataset = sdf[['dt1', 'dt2']].values

        model = gmm.GaussianMixture(n_components=n_components,
                covariance_type='full', max_iter=300, n_init=100, tol=1e-4,
                random_state=state, verbose=1)
        model.fit(dataset)

        self['n_components'] = n_components
        self.save_array('mixture/weights', model.weights_, title='Mixture '
                            'Component Weights')
        self.save_array('mixture/means', model.means_, title='Mixture Means')
        self.save_array('mixture/covariances', model.covariances_, title=''
                            'Mixture Covariances')

    def pdf_args(self, W, Mu, Cov):
        """Pre-compute arguments for sampling the pdf."""
        invC = np.array([la.inv(C) for C in Cov])
        detC = np.array([la.det(C) for C in Cov])
        Wnorm = (W / (2 * np.pi * np.sqrt(detC))).reshape((-1,1))
        args = Wnorm, Mu, invC, detC
        return args

    def pdf(self, X, Wnorm, Mu, invC, detC):
        """Evaluate the gaussian mixture probability density at values X."""
        Xbar = np.reshape(X, (1,-1,2)) - np.reshape(Mu, (-1,1,2))
        Pcomps = np.exp(-0.5 * np.sum((Xbar @ invC) * Xbar, axis=2))
        P = (Wnorm * Pcomps).sum(axis=0)

        pty.debug(f'invC.shape = {invC.shape}')
        pty.debug(f'detC.shape = {detC.shape}')
        pty.debug(f'Wnorm.shape = {Wnorm.shape}')
        pty.debug(f'Xbar.shape = {Xbar.shape}')
        pty.debug(f'Pcomps.shape = {Pcomps.shape}')
        pty.debug(f'P.shape = {P.shape}')
        return P

    @step
    def plot_mixture_model(self, nevals=128, levels=12, cmap='viridis',
        cscale='min', figsize=(6.625,3.14965)):
        """Plot figure of the gaussian mixture model with data points."""
        pty.debug_mode(True)
        sdf = self.read_dataframe('estimation/samples')
        w = self.read_array('mixture/weights')
        mu = self.read_array('mixture/means')
        cov = self.read_array('mixture/covariances')
        args = self.pdf_args(w, mu, cov)

        # Compute joint/conditional distros for comparison to model
        edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        H, _, _ = np.histogram2d(sdf.dt1, sdf.dt2, bins=edges)
        C = H / np.trapz(H, centers).reshape((-1,1))

        # Evaluate the model probabilty density on a coordinate grid
        x = np.linspace(self.c.dmin, self.c.dmax, nevals)
        X1, X2 = np.meshgrid(x, x)
        XX = np.c_[X1.ravel(), X2.ravel()]
        P = self.pdf(XX, *args)
        PP = P.reshape(X1.shape)
        modelP = PP / np.trapz(PP, x)

        # Set the shared colorscale max for the plots
        if cscale == 'data':
            vmax = C.max()
        elif cscale == 'model':
            vmax = modelP.max()
        elif cscale == 'min':
            vmax = min(C.max(), modelP.max())
        elif cscale == 'max':
            vmax = max(C.max(), modelP.max())
        else:
            vmax = None
        self.out(f'vmax = {vmax}', debug=True)

        f = self.figure(f'mixture-model-{self.c.n_components}-components',
                clear=True, figsize=figsize, title='Gaussian Mixture Model '
                    f'Density: N = {self.c.n_components}')
        axd = f.add_subplot(121)
        axm = f.add_subplot(122)

        def plot_distro(ax, P):
            im = ax.imshow(P.T, cmap=cmap, aspect='auto', vmin=0.0, vmax=vmax,
                    extent=[self.c.dmin, self.c.dmax] * 2)
            f = ax.get_figure()
            f.colorbar(im, ax=ax, shrink=0.8)
        plot_distro(axd, C)

        def plot_model_mesh(ax):
            m = ax.pcolormesh(X1, X2, modelP, cmap=cmap, vmin=0.0, vmax=vmax)
            f = ax.get_figure()
            f.colorbar(m, ax=ax, shrink=0.8)
        plot_model_mesh(axm)

        def set_axes(ax, title, show_ylabel=True):
            ra.quicktitle(ax, title, size='small', weight='light')
            ra.despine(ax)
            ax.set_xlim(self.c.dmin, self.c.dmax)
            ax.set_ylim(self.c.dmin, self.c.dmax)
            ax.axis('scaled')
            ax.set_xlabel('First normalized ISI')
            if show_ylabel:
                ax.set_ylabel('Second normalized ISI')
        set_axes(axd, 'Data Conditional')
        set_axes(axm, 'Model Conditional', show_ylabel=False)

    @step
    def evaluate_conditional(self, nevals=128, figsize=(9,4.5)):
        """Evaluate conditional densities in comparison with data."""
        sdf = self.read_dataframe('estimation/samples')
        w = self.read_array('mixture/weights')
        mu = self.read_array('mixture/means')
        cov = self.read_array('mixture/covariances')
        args = self.pdf_args(w, mu, cov)

        x = np.linspace(self.c.dmin, self.c.dmax, nevals)
        pdfs = np.empty((self.c.bins, x.size), 'd')
        histos = np.empty((self.c.bins, self.c.bins), 'd')

        edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        binwidth = (self.c.dmax - self.c.dmin) / self.c.bins

        for i in range(self.c.bins):
            data = sdf.loc[sdf.bin == i, 'dt2']
            xx = np.c_[np.zeros(nevals) + centers[i], x]
            pdfs[i,:] = self.pdf(xx, *args)
            histos[i,:], _ = np.histogram(data, bins=edges)

        f = self.figure('conditional-mixture', clear=True, figsize=figsize,
                title=f'Conditional Mixture Density: {self.c.phase.title()} '
                        'Cells')
        ax = f.add_subplot(111)

        cols = sns.husl_palette(n_colors=self.c.bins, h=0.4, s=0.8, l=0.6)

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

    @step
    def validation_sample(self, seed='wake-sleep', burn_samples=100,
        nevals=128, maxpad=0.05):
        """Sample the synthetic timing model and check fit to data sample."""
        state = np.random.RandomState(seed=sum(map(ord, seed)))
        w = self.read_array('mixture/weights')
        mu = self.read_array('mixture/means')
        cov = self.read_array('mixture/covariances')
        args = self.pdf_args(w, mu, cov)

        a, b = self.c.dmin, self.c.dmax
        x = np.linspace(self.c.dmin, self.c.dmax, nevals)

        nsamples = self.c.N_deltas
        samples = np.empty((nsamples,), 'd')

        pty.debug_mode(False)
        delta = -0.0557086  # beta-estimated marginal mode

        for i in range(-burn_samples, nsamples):
            evals = np.c_[np.zeros(nevals) + delta, x]
            pmax = self.pdf(evals, *args).max()

            while True:
                u1, u2 = state.random_sample(size=2)
                xval = (b - a) * u1 + a
                probe = (1 + maxpad) * pmax * u2
                density = self.pdf([delta, xval], *args)
                if probe <= density:
                    break

            delta = xval
            if i >= 0:
                samples[i] = delta
                self.box(filled=True, color='brown')
                continue
            self.box(filled=False, color='brown')
        self.newline()

        self.save_array('validation/samples', samples, title='Synthetic '
                f'Timing Samples: N = {nsamples}')

    @step
    def plot_validation_sample(self, cscale='min', samplebins=128,
        cmap='viridis', figsize=(6.625,3.14965), marg_figsize=(9,3.2)):
        """Plot figure of the validation sample with data for comparison."""
        sdf = self.read_dataframe('estimation/samples')
        data_samples = np.r_[sdf.dt1, sdf.dt2.iloc[-1]]
        val_samples = self.read_array('validation/samples')
        n_samples = data_samples.size
        assert (val_samples.size == n_samples), 'validation sample size ' \
             f'mismatch ({n_samples} != {val_samples.size})'

        edges = np.linspace(self.c.dmin, self.c.dmax, self.c.bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        def construct_distros(data):
            dt1 = data[:-1]
            dt2 = data[1:]
            H, _, _ = np.histogram2d(dt1, dt2, bins=edges)
            joint = H / np.trapz(np.trapz(H, centers), centers)
            cond = H / np.trapz(H, centers).reshape((-1,1))
            return joint, cond
        Jdata, Cdata = construct_distros(data_samples)
        Jval, Cval = construct_distros(val_samples)

        # Set the shared colorscale max for the plots
        if cscale == 'data':
            vmax = Cdata.max()
        elif cscale == 'model':
            vmax = Cval.max()
        elif cscale == 'min':
            vmax = min(Cdata.max(), Cval.max())
        elif cscale == 'max':
            vmax = max(Cdata.max(), Cval.max())
        else:
            vmax = None
        self.out(f'vmax = {vmax}', debug=True)

        #
        # Create a 2x2 figure showing (data, validation) x (conditional, joint)
        #

        f = self.figure('mixture-validation', clear=True, figsize=figsize,
                title='Data (left) and Model Validation Sample (right): ' +
                    f'n = {n_samples}')
        axdc = f.add_subplot(121)   # data conditional
        axvc = f.add_subplot(122)   # model (sample) conditional

        def plot_distro(ax, P):
            im = ax.imshow(P.T, cmap=cmap, aspect='auto', vmin=0, vmax=vmax,
                    extent=[self.c.dmin, self.c.dmax] * 2)
            f = ax.get_figure()
            f.colorbar(im, ax=ax, shrink=0.8)
        plot_distro(axdc, Cdata)
        plot_distro(axvc, Cval)

        def set_axes(ax, title, show_ylabel=True):
            ra.quicktitle(ax, title, size='small', weight='light')
            ra.despine(ax)
            ax.set_xlim(self.c.dmin, self.c.dmax)
            ax.set_ylim(self.c.dmin, self.c.dmax)
            ax.axis('scaled')
            ax.set_xlabel('First normalized ISI')
            if show_ylabel:
                ax.set_ylabel('Second normalized ISI')
        set_axes(axdc, 'Data Conditional')
        set_axes(axvc, 'Validation Conditional', show_ylabel=False)

        return

        #
        # Create a 1x3 figure showing data/validation samples and marginals
        #

        def construct_marginal(data):
            H, _ = np.histogram(data, bins=edges)
            marg = H / np.trapz(H, centers)
            return marg
        Mdata = construct_marginal(data_samples)
        Mval = construct_marginal(val_samples)

        f = self.figure('mixture-marginal', clear=True, figsize=marg_figsize,
                title='Marginal Timing for Data and Mixture Model')
        axd = f.add_subplot(131)
        axv = f.add_subplot(132, sharey=axd)
        axc = f.add_subplot(133, sharey=axd)

        f.subplots_adjust(wspace=0.18, bottom=0.17)
        edges = np.linspace(self.c.dmin, self.c.dmax, samplebins + 1)
        cols = ['slateblue', 'forestgreen']

        def plot_marg_samples(ax, marg, data, title, col, ylabel=True,
            legend=False):
            ax.hist(data, bins=edges, normed=True, color=col, alpha=0.8,
                    zorder=-1, label='Samples')
            ax.plot(centers, marg, c='k', ls=':', lw=1.5, label='Marginal')
            ax.set_xlabel('Normalized timing')
            if ylabel:
                ax.set_ylabel('Pr[timing]')
            ax.set_ylim(bottom=0.0)
            ra.quicktitle(ax, title, size='medium', weight='light')
            ra.despine(ax)
            if legend:
                ax.legend()
        plot_marg_samples(axd, Mdata, data_samples, 'Data', cols[0],
                legend=True)
        plot_marg_samples(axv, Mval, val_samples, 'Model validation sample',
                cols[1], ylabel=False)

        def plot_comparo(ax, marg, label, col):
            ax.plot(centers, marg, c=col, ls='-', lw=1.8, alpha=0.7,
                    label=label)
        plot_comparo(axc, Mdata, 'Data', cols[0])
        plot_comparo(axc, Mval, 'Model', cols[1])

        def finish_comparo(ax):
            ax.set(xlabel='Normalized timing')
            ra.despine(ax)
            ax.legend()
        finish_comparo(axc)

    @step
    def plot_timing_distros(self, figsize=(12,8)):
        """Plot figure of the raw and normalized 2D timing distributions."""
        if self.c.phase == 'day':
            tlim = (-3.5, 0.12)
        else:
            tlim = (-2, -0.5)

        # Collate raw ISI samples
        raw_samples = None
        for label in self.c.labels:
            ts = patch.PatchSpikes(label).t
            dt = np.log(np.diff(ts))
            dt_pairs = np.c_[dt[:-1], dt[1:]]

            if raw_samples is None:
                raw_samples = dt_pairs
            else:
                raw_samples = np.concatenate((raw_samples, dt_pairs))

        # Load conditional normalized samples
        sdf = self.read_dataframe('estimation/samples')
        data_samples = np.c_[sdf.dt1, sdf.dt2]
        dlim = (self.c.dmin, self.c.dmax)

        def construct_distros(data, tmin, tmax):
            edges = np.linspace(tmin, tmax, self.c.bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            dt1 = data[:,0]
            dt2 = data[:,1]
            H, _, _ = np.histogram2d(dt1, dt2, bins=edges)
            joint = H / np.trapz(np.trapz(H, centers), centers)
            cond = H / np.trapz(H, centers).reshape((-1,1))
            return joint, cond
        Jraw, Craw = construct_distros(raw_samples, tlim[0], tlim[1])
        Jdata, Cdata = construct_distros(data_samples, dlim[0], dlim[1])

        #
        # Create a 2x2 figure showing (raw, normalized) x (joint, conditional)
        #

        # sns.set_context('talk')
        # mpl.rcParams['axes.labelpad'] = 3.3

        f = self.figure('timing-distros', clear=True, figsize=figsize,
                title='Pooled ISI (left) and Rate-Normalized (right) Log Timing '
                        f'Distributions (n={raw_samples.shape[0]} pairs)')
        axrj = f.add_subplot(221)   # raw joint
        axdj = f.add_subplot(222)   # data joint
        axrc = f.add_subplot(223)   # raw conditional
        axdc = f.add_subplot(224)   # data conditional

        f.subplots_adjust(wspace=0.36, hspace=0.23, left=0.15, right=0.85,
                bottom=0.1)
        cbargs = dict(shrink=0.7, pad=0.03, aspect=9)

        def plot_distro(ax, P, title, lim, cmap, xlabel, ylabel):
            im = ax.imshow(P.T, vmin=0.0, aspect='equal', cmap=cmap,
                    extent=lim * 2)
            ax.axis('auto')
            plt.colorbar(mappable=im, ax=ax, **cbargs)
            if xlabel:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ra.quicktitle(ax, title, size='small')
            ra.despine(ax)
        plot_distro(axrc, Craw, r'ISI, P($\Delta{t_2}|\Delta{t_1}$)', tlim,
                'viridis', r'Log $\Delta{t_1}$', r'Log $\Delta{t_2}$')
        plot_distro(axdc, Cdata, r'Normalized, P($\Delta{t_2}|\Delta{t_1}$)',
                dlim, 'viridis', r'Normalized Log $\Delta{t_1}$',
                r'Normalized Log $\Delta{t_2}$')
        plot_distro(axrj, Jraw, r'ISI, P($\Delta{t_1},\Delta{t_2}$)', tlim,
                'viridis', None, r'Log $\Delta{t_2}$')
        plot_distro(axdj, Jdata, r'Normalized, P($\Delta{t_1},\Delta{t_2}$)',
                dlim, 'viridis', None, r'Normalized Log $\Delta{t_2}$')

        sns.reset_orig()

    @step
    def fit_rate_distro(self, pad=0.1, nevals=128, figsize=(5,4)):
        """Fit a beta distribution to average firing rate in the data."""
        rates = np.array([patch.PatchSpikes(label).t.size for label in
                    self.c.labels]) / DURATION

        padding = 0.5 * pad * rates.ptp()
        loc = rates.min() - padding
        scale = rates.ptp() + 2 * padding
        a, b, _, _ = st.beta.fit(rates, floc=loc, fscale=scale)

        # Save the parameters for sampling
        self['a_rate'] = a
        self['b_rate'] = b
        self['loc_rate'] = loc
        self['scale_rate'] = scale

        # Evaluate beta pdf for plotting
        xmin, xmax = st.beta.ppf([0.01, 0.99], a, b, loc=loc, scale=scale)
        x = np.linspace(xmin, xmax, nevals)
        pdf = st.beta.pdf(x, a, b, loc=loc, scale=scale)

        f = self.figure('rate-distro', clear=True, figsize=figsize,
                title=r'Mean Firing Rate $\beta$ Distribution: '
                    f'{self.c.phase.title()} Cells')
        ax = f.add_subplot(111)

        def plot_rates(ax):
            ax.vlines(rates, 0.0, pdf.max() / 3, label='Cell means')
            ax.plot(x, pdf, c='turquoise', lw=2, zorder=-1,
                    label=r'$\beta$ density')
            ax.set(xlabel='Firing rate (spike/s)', ylabel='Pr[rate]')
            ax.set_ylim(bottom=0.0)
            ax.legend()
            ra.despine(ax)
        plot_rates(ax)

    @step
    def generate_spike_trains(self, n=10, duration=DURATION, firing_rate=None,
        burn_samples=100, nevals=128, maxpad=0.05, seed='wake-sleep',
        ret_trains=False, plot=True, figsize=(8.5,5)):
        """Generate synthetic spike trains by sampling the mixture model."""
        state = np.random.RandomState(seed=sum(map(ord, seed)))
        w = self.read_array('mixture/weights')
        mu = self.read_array('mixture/means')
        cov = self.read_array('mixture/covariances')
        args = self.pdf_args(w, mu, cov)
        pty.debug_mode(False)

        a, b = self.c.dmin, self.c.dmax
        x = np.linspace(self.c.dmin, self.c.dmax, nevals)

        beta_rate = st.beta(self.c.a_rate, self.c.b_rate, loc=self.c.loc_rate,
                scale=self.c.scale_rate)

        rates_are_fixed = False
        if firing_rate is not None:
            try:
                fixed_rates = float(firing_rate) * np.ones(n)
            except TypeError:
                assert len(firing_rate) == n, 'firing_rate size mismatch: ' \
                    f'{len(firing_rate)} != {n}'
                fixed_rates = firing_rate
            rates_are_fixed = True

        trains = []
        mean_rates = []
        for j in range(n):
            if rates_are_fixed:
                rate = fixed_rates[j]
            else:
                rate = beta_rate.rvs(random_state=state)
            mean_rates.append(rate)
            nspikes = int(np.ceil(rate * duration))
            nsamples = nspikes + 1  # n-1 intra-train, +2 at start/end
            samples = np.empty((nsamples,), 'd')

            self.out(f'Generating train {j} at {rate:.3f} spikes/s...')
            delta = -0.0557086  # beta-estimated marginal mode
            for i in range(-burn_samples, nsamples):
                evals = np.c_[np.zeros(nevals) + delta, x]
                pmax = self.pdf(evals, *args).max()

                while True:
                    u1, u2 = state.random_sample(size=2)
                    xval = (b - a) * u1 + a
                    probe = (1 + maxpad) * pmax * u2
                    density = self.pdf([delta, xval], *args)
                    if probe <= density:
                        break

                delta = xval
                if i >= 0:
                    samples[i] = delta

            # Transform and normalize log-intervals into spike times
            t = np.cumsum(np.exp(samples))
            newtrain = ((duration / t[-1]) * t)[:-1]
            trains.append(newtrain)

        if plot:
            self.plot_generated_spike_trains(mean_rates, trains,
                    duration=duration, figsize=figsize)

        if ret_trains:
            return mean_rates, trains

    def plot_generated_spike_trains(self, rates, trains, duration=DURATION,
        figsize=(8,4)):
        """Create a figure showing the newly generated spike trains."""
        n = len(trains)
        f = self.figure(f'sample-{self.c.phase}-trains', clear=True,
                figsize=figsize, title=f'Synthetic \'{self.c.phase.title()}\' '
                    f'Spike Trains: N = {n}')
        ax = f.add_subplot(111)

        f.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.92)
        time_pad = 0.03
        raster_pad = 0.1

        def plot_raster(ax, i, st, rate):
            ax.vlines(st, i + raster_pad, i + 1 - raster_pad, linewidths=0.6,
                    colors='k', alpha=0.8)
            ax.text(0, i + 1 - raster_pad, f'{rate:.1f} spk/s',
                    verticalalignment='top', horizontalalignment='left',
                    fontdict={'size':'x-small', 'weight':'light'},
                    bbox={'facecolor':'w', 'edgecolor':'none', 'alpha':0.8})

        for j in range(n):
            plot_raster(ax, j, trains[j], rates[j])

        def finish_plot(ax):
            ax.set_xlim(-time_pad*duration, (1+time_pad) * duration)
            ax.set_ylim(-raster_pad, n + raster_pad)
            ax.set_yticks(np.r_[0:n] + 0.5)
            ax.set_yticklabels(np.arange(1, n+1))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Synthetic spike trains')
            ra.despine(ax)
        finish_plot(ax)
