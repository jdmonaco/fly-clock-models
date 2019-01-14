"""
Detect and analyze spikes in fly wake/sleep voltage trace data.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import roto.axis as ra
import roto.data as rd

from .. import store
from ..context import FlyAnalysis, step
from ..lib import activity, data, patch
from ..tools import binned


DURATION = 60.0    # s


class SpikeAnalysis(FlyAnalysis):

    """
    Validate spike detection and perform event-triggered and interval analysis.
    """

    def cleanup(self):
        store.close()

    @step
    def plot_trace_histos(self, bins=128, figsize=(8, 6)):
        """Plot figure of voltage trace histograms for all recordings."""
        rdf = data.dataframe_from_table('/recordings')
        day = rdf.phase == 'day'
        night = rdf.phase == 'night'
        N = max(day.sum(), night.sum())

        f, axlist = plt.subplots(N, 2, sharex=True, sharey=True,
                num='trace-histos', figsize=figsize)
        self.figure('trace-histos', clear=False, handle=f,
                title='Voltage Histograms')

        def plot_histo(ax, v, col):
            ax.hist(v, bins=bins, color=col)
            ra.despine(ax)

        # Plot voltage histograms
        cols = 'b', 'r'
        for j, phase in enumerate(('night', 'day')):
            for i, rec in enumerate(rdf.loc[rdf.phase == phase].itertuples()):
                pt = patch.PatchTrace(rec.label)
                ax = axlist[i,j]
                plot_histo(ax, pt.v, cols[j])

                if i == 0:
                    if j == 0:
                        ax.set_ylabel('Samples')
                    ax.set_title(phase.title())
                if i == N - 1:
                    ax.set_xlabel('Voltage (mV)')

        # Compute median baseline differences
        baseline = dict()
        for j, phase in enumerate(('night', 'day')):
            vm = []
            for i, rec in enumerate(rdf.loc[rdf.phase == phase].itertuples()):
                pt = patch.PatchTrace(rec.label)
                vm.append(np.median(pt.v))
            vm = np.array(vm)
            self.out(f'Median {phase} voltage = {vm.mean():.2f} ± '
                     f'{vm.std():.2f} mV')
            baseline[phase] = vm

    @step
    def detect_spikes(self, h=activity.SPIKE_HWIN, dv=activity.SPIKE_DV):
        """Perform spike detection on all recordings and save to database."""
        f = store.get(False)
        rdf = data.dataframe_from_table('/recordings', index='id')

        for rec in rdf.itertuples():
            self.out('Finding spikes for cell {}...', rec.label)
            P = patch.PatchTrace(rec.label)
            spikes = activity.find_spikes(P.t, P.v, half_window=h,
                    dv_thresh=dv, return_voltage=True)
            spikes = np.array(spikes)
            self.out('Found {} spikes!', len(spikes))

            path = os.path.join(rec.path, 'spikes')
            tarr = rd.new_array(f, path, 'T', spikes[:,0], createparents=True)
            self.out('Saved: {}', tarr._v_pathname)
            varr = rd.new_array(f, path, 'V', spikes[:,1])
            self.out('Saved: {}', varr._v_pathname)

        self.out('All done!')

    @step
    def plot_detected_spikes(self, figsize=(8, 6)):
        """Plot detected spikes overlaid on voltage trace for validation."""
        rdf = data.dataframe_from_table('/recordings', index='id')
        base = self.mkdir('spikes')
        self.set_figfmt('png')

        for rec in rdf.itertuples():
            trace = patch.PatchTrace(rec.label)
            spikes = patch.PatchSpikes(rec.label)

            figlabel = 'spikes-{}-{}'.format(rec.phase, rec.label)
            f = self.figure(figlabel, clear=True, figsize=figsize,
                    title='Detected Spikes: {}'.format(rec.label))
            ax = f.add_subplot(111)

            ax.plot(trace.t, trace.v, c='b', lw=0.6, label='V')
            ax.plot(spikes.t, spikes.v, c='r', ls='', marker='x', mew=1,
                    label='spikes')
            ax.set(xlabel='Time (s)', ylabel='Voltage (mV)')
            ax.legend(loc='upper right')
            ra.despine(ax)

            self.savefig(basepath=os.path.join(base, figlabel))

    @step
    def plot_isi_distros(self, ymax=1.5, figsize=(8,6)):
        """Plot interspike interval distributions."""
        rdf = data.dataframe_from_table('/recordings', index='id')

        def plot_isi_figure(phase, col):
            f = self.figure('isi-distro-{}'.format(phase), clear=True,
                    figsize=figsize, title='ISI Distributions: '
                        '{}'.format(phase.title()))
            ax = f.add_subplot(111)

            labels = rdf.loc[rdf.phase == phase, 'label']
            data = [np.diff(patch.PatchSpikes(label).t) for label in labels]
            N = len(labels)

            vp = ax.violinplot(data, showmedians=True)
            [body.set_color(col) for body in vp['bodies']]
            [vp[k].set_color(col) for k in (
                    'cmins', 'cmaxes', 'cbars', 'cmedians')]

            ax.set_xticks(np.arange(1, N+1))
            ax.set_xticklabels(labels, rotation=45, size='x-small',
                fontweight='light')
            ax.set(xlabel='Cells', ylabel='ISI (s)')
            ax.set_ylim(0.0, ymax)
            ra.despine(ax)

        plot_isi_figure('night', 'b')
        plot_isi_figure('day', 'r')

    @step
    def plot_spike_voltage_traces(self, h=0.02, figsize=(8,6)):
        """Plot spike-triggered voltage traces and spike-triggered averages."""
        rdf = data.dataframe_from_table('/recordings', index='id')
        N = max(np.sum(rdf.phase == 'night'), np.sum(rdf.phase == 'day'))
        path = self.mkdir('spike-shapes')

        traces = {}
        delta = None
        for phase in ('night', 'day'):
            for label in rdf.loc[rdf.phase == phase, 'label']:
                self.out('Computing {} cell {}...', phase, label)
                P = patch.PatchTrace(label)
                S = patch.PatchSpikes(label)
                dt, tr = activity.spike_traces(S.t, P.t, P.v, half_window=h)
                traces[label] = tr
                if delta is None:
                    delta = dt

        self.out('Plotting figure...')
        plt.figure('spike-shapes').clear()
        f, ax = plt.subplots(N, 2, sharex=True, sharey=True,
                num='spike-shapes', figsize=figsize)
        self.figure('spike-shapes', clear=False, handle=f,
                title='Spike-Triggered Voltage Traces')

        def plot_spike_traces(ax, tr, col):
            ax.plot(delta, tr, c=col, ls='-', lw=0.5, alpha=0.2, zorder=0)
            ax.plot(delta, tr.mean(axis=1), c='k', ls='-', lw=1.3,
                    alpha=0.8, zorder=1)
            ax.axvline(ls=':', c='k', lw=0.8, zorder=2)

        def plot_figure(phase, label, tr, col):
            f = self.figure('spike-shapes-{}'.format(label), figsize=figsize,
                    title='Spike Shapes: {} [{}]'.format(label, phase))
            ax = f.add_subplot(111)
            plot_spike_traces(ax, tr, col)
            ax.set_xlabel(r'$\Delta{t}$ (s)')
            ax.set_ylabel('Voltage (mV)')
            self.savefig(basepath=os.path.join(path,
                'cell-{}-{}'.format(phase, label)), closeafter=True)

        cols = 'b', 'r'
        for j, phase in enumerate(['night', 'day']):
            for i, label in enumerate(rdf.loc[rdf.phase == phase, 'label']):
                plot_spike_traces(ax[i,j], traces[label], cols[j])
                plot_figure(phase, label, traces[label], cols[j])
                if i == 0:
                    ax[i,j].set_title(phase.title())
                elif i == N - 1:
                    ax[i,j].set_xlabel(r'$\Delta{t}$ (s)')
                if i == 0 and j == 0:
                    ax[i,j].set_ylabel('Voltage (mV)')

        f = self.figure('spike-shape-averages', clear=True, figsize=figsize,
                title='Spike-Triggered Voltage Averages: All Cells')
        ax = f.add_subplot(111)
        df = store.get(False)

        self.out('Computing grand within-phase averages...')
        for j, phase in enumerate(['night', 'day']):
            sta = None
            for i, label in enumerate(rdf.loc[rdf.phase == phase, 'label']):
                trmean = traces[label].mean(axis=1)
                if sta is None:
                    sta = trmean.reshape(-1,1)
                else:
                    sta = np.concatenate((sta, trmean[:,np.newaxis]), axis=1)

            ax.plot(delta, sta, c=cols[j], ls='-', lw=0.9, alpha=0.4, zorder=0)
            ax.plot(delta, sta.mean(axis=1), c=cols[j], ls='-', lw=2.5,
                    alpha=0.9, zorder=1, label='{} avg.'.format(phase.title()))

            if j == 0:
                rd.new_array(df, '/spike_shape', 'delta_t', delta,
                        title='Time Lags for Spike-Triggered Voltage Traces',
                        createparents=True)
            rd.new_array(df, '/spike_shape', '{}_avg'.format(phase),
                    sta.mean(axis=1), title='Spike-Triggered Voltage Trace '
                        'Average: {}'.format(phase.title()))

        ax.set_xlabel(r'$\Delta{t}$ (s)')
        ax.set_ylabel('Voltage (mV)')
        ax.legend(loc='upper right')
        ra.despine(ax)

    @step
    def compute_timing(self, kernel_width=activity.KERNEL_WIDTH):
        """Compute timing and rate estimates for model comparison."""
        df = store.get(False)
        rdf = data.dataframe_from_table('/recordings', index='id')
        rd.new_group(df, '/', 'timing', title='Timing Estimates')

        def run_dist(rec, st):
            self.out('Computing {} cell {}...', rec.phase, rec.label)
            Pdt = binned.jointtiming(st)
            arr = rd.new_array(df, '/timing', 'dist_{}'.format(rec.label), Pdt,
                    title='Joint Diff-Diff Timing Distribution: '
                        '{}'.format(rec.label))
            self.out('Saved: {}', arr._v_pathname)

        def run_rate(rec, st, tt):
            estimator = activity.FiringRateEstimate(st, DURATION,
                    width=kernel_width, Fs_norm=100.0)
            Rest = estimator.evaluate(tt)
            arr = rd.new_array(df, rec.path, 'rate_slow', Rest,
                    title='Estimated Firing Rate: {}'.format(rec.label))
            arr._v_attrs['kernel_width'] = kernel_width
            self.out('Saved: {}', arr._v_pathname)

        def run_lowpass(rec, v):
            vlp = activity.lowpass_filter(v)
            arr = rd.new_array(df, rec.path, 'V_lowpass', vlp,
                    title='Low-Pass Filtered Voltage: {}'.format(rec.label))
            self.out('Saved: {}', arr._v_pathname)

        for rec in rdf.itertuples():
            trace = patch.PatchTrace(rec.label)
            spk = patch.PatchSpikes(rec.label)
            run_dist(rec, spk.t)
            run_rate(rec, spk.t, trace.t)
            run_lowpass(rec, trace.v)

    @step
    def plot_timing_distros(self, figsize=(9, 3.5)):
        """Plot figure of joint diff-diff timing distributions."""
        df = store.get(False)
        rdf = data.dataframe_from_table('/recordings', index='id')
        day = rdf.phase == 'day'
        night = rdf.phase == 'night'
        N = max(day.sum(), night.sum())

        mpl.rcParams['figure.subplot.wspace'] = 0.12
        mpl.rcParams['figure.subplot.hspace'] = 0.12

        plt.figure('timing-distros').clear()
        f, axlist = plt.subplots(2, N, sharex=True, sharey=True,
                num='timing-distros', figsize=figsize)
        self.figure('timing-distros', clear=False, handle=f)

        tmin, tmax = binned.TIME_MIN, binned.TIME_MAX
        extent = [tmin, tmax, tmin, tmax]

        def plot_distro(ax, P):
            ax.imshow(P, cmap='viridis', vmin=0, extent=extent, aspect='equal')
            ax.axis('auto')
            ra.despine(ax)

        for i, phase in enumerate(('night', 'day')):
            for j, rec in enumerate(rdf.loc[rdf.phase == phase].itertuples()):
                P = df.get_node('/timing', 'dist_{}'.format(rec.label)).read()
                ax = axlist[i,j]
                plot_distro(ax, P)

                if j == 0:
                    ax.set_ylabel(phase.title())
                if i == 1:
                    ax.set_xlabel(r'$\Delta{t}$ (s)')

        f.tight_layout()
        mpl.rc_file_defaults()

        f = self.figure('timing-distros-avg', clear=True, figsize=(8,4.5),
                title='Joint Second-Order Spike Timing Distributions: '
                    'Cell Average')

        for i, phase in enumerate(('night', 'day')):
            Pavg = None
            for j, rec in enumerate(rdf.loc[rdf.phase == phase].itertuples()):
                P = df.get_node('/timing', 'dist_{}'.format(rec.label)).read()
                if Pavg is None:
                    Pavg = P
                else:
                    Pavg += P

            ax = f.add_subplot(1,2,1+i)
            plot_distro(ax, Pavg)
            ax.set_title(phase.title())
            if i == 0:
                ax.set(ylabel=r'$\Delta{t_2}$ (s)')
            ax.set(xlabel=r'$\Delta{t_1}$ (s)')

            arr = rd.new_array(df, '/timing', 'dist_{}_avg'.format(phase),
                    Pavg, title='Summed Joint Timing Distribution: '
                        '{}'.format(phase.title()))
            self.out('Saved: {}', arr._v_pathname)

    @step
    def plot_slow_estimates(self, figsize=(8,6)):
        """Plot figure of slow kernel-based firing rate estimates."""
        rdf = data.dataframe_from_table('/recordings', index='id')
        df = store.get()
        day = rdf.phase == 'day'
        night = rdf.phase == 'night'
        N = max(day.sum(), night.sum())

        figname = 'slow-estimates'
        f, axlist = plt.subplots(N, 2, sharex=True, sharey=True,
                num=figname, figsize=figsize)
        self.figure(figname, clear=False, handle=f,
                title='Rate ({:.1f}-s) vs. Lowpass-V ({:.1f}-s) '
                    'Estimates'.format(activity.KERNEL_WIDTH,
                        activity.LOWPASS_TAU))

        def plot_estimates(ax, pt):
            ax.plot(pt.t, pt.r, c='k')
            axv = ax.twinx()
            axv.plot(pt.t, pt.v_slow, c='b')
            ra.despine(ax)
            axv.spines['left'].set_visible(False)
            axv.spines['top'].set_visible(False)

        for j, phase in enumerate(('night', 'day')):
            for i, rec in enumerate(rdf.loc[rdf.phase == phase].itertuples()):
                ax = axlist[i,j]
                plot_estimates(ax, patch.PatchTrace(rec.label))

                if i ==0 and j == 0:
                    ax.set_ylabel(r'$\hat{R}$ (spk/s)')
                if i == 0:
                    ax.set_title(phase.title())
                elif i == N - 1:
                    ax.set_xlabel(r'$\Delta{t}$ (s)')
