"""
Diurnal simulations of a circadian-controlled neuron.
"""

from brian2 import *
import matplotlib as mpl

from roto.arrays import discretize
import roto.axis as ra

from ..context import FlySimulation, step
from ..lib.sim import triangle_wave
from ..lib.activity import spike_traces, find_spikes


R = 8.3144598        # Ideal gas constant
T = 298.15           # 24 ºC
F = 96485.33289      # Faraday's constant
RT_F = 1000.0*R*T/F  # Nernst potential scaling factor


class ClockNeuronModel(FlySimulation):

    """
    Single-neuron models of diurnal modulation of firing patterns.
    """

    @step
    def baseline_tuning(self, N=10, deltalim=(0.0,1.0), v0=-55.0,
        duration=0.3):
        """Baseline resting membrane potential tuning of SCN model."""
        self['tuning_N'] = N
        self['tuning_range'] = deltalim
        self['tuning_dur'] = duration

        defaultclock.dt = 0.1*ms
        dur = duration*second

        # Conductance parameters
        Cm = 5.7 * pF
        gNa = 150 * nS  # 229 * nS
        gK = 14 * nS
        gCa = 65 * nS
        gCl = 0.3 * nS
        ECa = 64 * mV
        ECl = -60 * mV
        ENAlcn = -20 * mV

        # Rhythmic conductance parameters
        gNAlcn_ZT0 = 0.22 * nS
        gNAlcn_delta = -0.1 * nS
        gKleak_ZT0 = 0.04 * nS
        gKleak_delta = 0.02 * nS

        # KCa (BK) conductance parameters
        gBK_day = 10 * nS
        gBK_delta = 10 * nS

        # Na+/K+ reversals for joint ATPase-based tuning
        ENa_scn = 45.0  # mV, reversal from Allada bicycle model
        Na_out = 140.0  # mM
        Na_in = Na_out / (np.exp(ENa_scn/RT_F))  # 24.3 mM
        EK_scn = -97.0  # ibid.
        K_out = 5.0  # mM
        K_in = K_out / (np.exp(EK_scn/RT_F))  # 218.1 mM

        eqns = """
            dv/dt = (-gNa*m**3*h*(v-ENa) - gNAlcn*(v-ENAlcn) - gK*n**4*(v-EK) - gKleak*(v-EK) - gCl*(v-ECl) - gCa*r*f*(v-ECa) - gBK*b*(v-EK)) / Cm : volt

            # Bicycle factors
            gNAlcn = gNAlcn_ZT0 + bicycle * gNAlcn_delta : siemens
            gKleak = gKleak_ZT0 + bicycle * gKleak_delta : siemens
            bicycle : 1 (constant)

            # Night factors
            EK : volt (constant)
            ENa : volt (constant)
            gBK : siemens (constant)

            # Channel activation dynamics
            dm/dt = (minf - m) / (exp(-(v/mV+286)/160)*ms) : 1
            minf = 1 / (1 + exp(-(v/mV+35.2)/8.1)) : 1
            dh/dt = (hinf - h) / ((0.51 + exp(-(v/mV+26.6)/7.1))*ms) : 1
            hinf = 1 / (1 + exp((v/mV+62)/4)) : 1
            dn/dt = (ninf - n) / (exp(-(v/mV-67)/68)*ms) : 1
            ninf = 1 / (1 + exp((v/mV-14)/(-17)))**0.25 : 1
            dr/dt = (rinf - r) / (3.1*ms) : 1
            rinf = 1 / (1 + exp(-(v/mV+25)/7.5)) : 1
            df/dt = (finf - f) / (exp(-(v/mV-444)/220)*ms) : 1
            finf = 1 / (1 + exp((v/mV+260)/65)) : 1
            db/dt = (binf - b) / (50*ms) : 1
            binf = 1 / (1 + exp(-(v/mV+20)/2)) : 1
        """
        G = NeuronGroup(2*N, model=eqns, threshold='v>-25*mV', refractory=5*ms,
                method='rk4', name='tuning_neuron')
        G.v = v0 * mV
        G.gBK[:N] = gBK_day
        G.gBK[N:] = gBK_day + gBK_delta
        G.bicycle = 0.5

        # Na/K reversal parameterization
        x = np.tile(np.linspace(deltalim[0], deltalim[1], N), 2)
        G.ENa = RT_F * np.log((Na_out + 3*x) / (Na_in - 3*x)) * mV
        G.EK = RT_F * np.log((K_out - 2*x) / (K_in + 2*x)) * mV
        self['tuning_ENa'] = list(G.ENa / mV)
        self['tuning_EK'] = list(G.EK / mV)
        self['tuning_x'] = list(x)

        mon = StateMonitor(G, ['v'], record=True, name='mon')

        net = Network(G, mon, name='tuning_network')
        net.run(dur)

        self.save_simulation('tuning', net)

    @step
    def plot_tuning(self, figsize=(13.06122449, 7.612244)):
        """Plot figure of baseline resting potential tuning."""
        S = self.read_simulation('tuning')
        mon = S.mon
        N = self.c.tuning_N
        dmin, dmax = self.c.tuning_range
        dur = self.c.tuning_dur

        f = self.figure('tuning', clear=True, figsize=figsize,
                title=f'Baseline RMP Tuning: 3:2 {dmin:.1f}-{dmax:.1f} mM')
        axr = f.add_subplot(231)
        axd = f.add_subplot(232)
        axe = f.add_subplot(233)
        axn = f.add_subplot(235, sharey=axd, sharex=axd)
        axo = f.add_subplot(236, sharey=axe, sharex=axe)

        cols = mpl.cm.GnBu(np.linspace(0,1,N))

        def plot_reversals(ax, xlabel=True):
            kwds = dict(c=cols)
            ax.scatter(self.c.tuning_x[:N], self.c.tuning_ENa[:N], label='Na+',
                    **kwds)
            ax.scatter(self.c.tuning_x[:N], self.c.tuning_EK[:N], label='K+',
                    **kwds)
            if xlabel:
                ax.set_xlabel('3:2 Concentration Change (mM)')
            ax.set_ylabel('Reversal potential (mV)')
            ax.legend(loc='center')
        plot_reversals(axr)

        def plot_traces(ax, start=0, xlabel=True):
            t = mon.loc[mon.neuron == 0, 't']
            for i in range(start, N+start):
                V = mon.loc[mon.neuron == i, 'v']
                ax.plot(t, 1000.0*V, c=cols[i%N], label=self.c.tuning_x[i])
            ax.set_xlim(0, dur)
            if xlabel:
                ax.set_xlabel('Time (s)')
            ax.set_ylabel(r'$V$ (mV)')
        plot_traces(axd, 0, xlabel=False)
        plot_traces(axn, N)

        def plot_rmp(ax, start=0, xlabel=True):
            RMP = []
            for i in range(start, N+start):
                lastV = mon.loc[mon.neuron == i, 'v'].values[-1]
                RMP.append(lastV)
            RMP = 1000.0 * np.array(RMP)
            X = self.c.tuning_x[start:N+start]
            ax.scatter(X, RMP, c=cols)
            if xlabel:
                ax.set_xlabel('3:2 Concentration Change (mM)')
        plot_rmp(axe, xlabel=False)
        plot_rmp(axo, start=N)

    def run(self, **kwds):
        """Script function to run simulation and plotting methods."""
        self.noise_simulation(**kwds)
        self.plot()
        self.plot_figure()

    @step
    def noise_simulation(self, day_app=0.0, day_noise=3.0, night_app=13.0,
        night_noise=3.0, duration=10.0, warmup=5.0, rngseed='irregularity'):
        """Cosine-diurnal modulation of intrinsic neuron parameters."""
        self['warmup'] = warmup
        self['dur'] = duration
        defaultclock.dt = 0.1*ms
        dur = (warmup + duration)*second
        seed(sum(list(map(ord, rngseed))))

        # Conductance parameters
        Cm = 5.7 * pF
        gK = 14 * nS
        gCa = 65 * nS
        gCl = 0.3 * nS
        ECa = 64 * mV
        ECl = -60 * mV
        ENAlcn = -20 * mV

        # Rhythmic conductance parameters
        gNAlcn_ZT0 = 0.22 * nS
        gNAlcn_delta = -0.1 * nS
        gKleak_ZT0 = 0.04 * nS
        gKleak_delta = 0.02 * nS

        # KCa (BK) conductance parameters
        gBK_day = 10 * nS
        gBK_night = 20 * nS

        # Na+/K+ baseline concentrations
        ENa_scn = 45.0  # mV, reversal from Allada bicycle model
        Na_out = 140.0  # mM
        Na_in = Na_out / (np.exp(ENa_scn/RT_F))  # 24.3 mM
        EK_scn = -97.0  # ibid.
        K_out = 5.0  # mM
        K_in = K_out / (np.exp(EK_scn/RT_F))  # 218.1 mM

        # Na+/K+ pump modulation (tuning parameters optimized from above)
        xday = -1.5
        xnight = 1.5
        ENa_day = RT_F * np.log((Na_out + 3*xday) / (Na_in - 3*xday)) * mV
        EK_day = RT_F * np.log((K_out - 2*xday) / (K_in + 2*xday)) * mV
        ENa_night = RT_F * np.log((Na_out + 3*xnight) / (Na_in - 3*xnight)) * mV
        EK_night = RT_F * np.log((K_out - 2*xnight) / (K_in + 2*xnight)) * mV

        # Report the calculate reversal potentials
        self.out('ENa (day) = {}', ENa_day)
        self.out('ENa (night) = {}', ENa_night)
        self.out('EK (day) = {}', EK_day)
        self.out('EK (night) = {}', EK_night)

        oup = Equations("""
            deta/dt = -eta/tau + xi*tau**-0.5 : 1
        """, tau=1*second)
        Noise = NeuronGroup(1, model=oup, method='euler', name='noise_source')

        eqns = """
            dv/dt = (Inoise + Iapp - gNa*m**3*h*(v-ENa) - gNAlcn*(v-ENAlcn) - gK*n**4*(v-EK) - gKleak*(v-EK) - gCl*(v-ECl) - gCa*r*f*(v-ECa) - gBK*b*(v-EK)) / Cm : volt

            # Bicycle factors
            gNAlcn = gNAlcn_ZT0 + bicycle * gNAlcn_delta : siemens
            gKleak = gKleak_ZT0 + bicycle * gKleak_delta : siemens
            bicycle : 1 (constant,shared)

            # Night factors
            EK : volt (constant)
            ENa : volt (constant)
            gBK : siemens (constant)
            gNa : siemens (constant)

            # Input currents: applied and slow noisy drive
            Iapp = app * pA : amp
            Inoise = sigma * eta * pA : amp
            eta : 1 (linked)
            sigma : 1 (constant)
            app : 1 (constant)

            # Channel activation dynamics
            dm/dt = (minf - m) / (exp(-(v/mV+286)/160)*ms) : 1
            minf = 1 / (1 + exp(-(v/mV+35.2)/8.1)) : 1
            dh/dt = (hinf - h) / ((0.51 + exp(-(v/mV+26.6)/7.1))*ms) : 1
            hinf = 1 / (1 + exp((v/mV+62)/4)) : 1
            dn/dt = (ninf - n) / (exp(-(v/mV-67)/68)*ms) : 1
            ninf = 1 / (1 + exp((v/mV-14)/(-17)))**0.25 : 1
            dr/dt = (rinf - r) / (3.1*ms) : 1
            rinf = 1 / (1 + exp(-(v/mV+25)/7.5)) : 1
            df/dt = (finf - f) / (exp(-(v/mV-444)/220)*ms) : 1
            finf = 1 / (1 + exp((v/mV+260)/65)) : 1
            db/dt = (binf - b) / (50*ms) : 1
            binf = 1 / (1 + exp(-(v/mV+20)/2)) : 1
        """
        G = NeuronGroup(4, model=eqns, threshold='v>-28*mV', refractory=5*ms,
                method='rk4', name='clock_neuron')
        G.v = -55.0 * mV
        G.bicycle = 0.5

        G.eta = linked_var(Noise.eta)
        G.sigma = [day_noise]*2 + [night_noise]*2
        G.app = [day_app]*2 + [night_app]*2

        G.gBK = [gBK_day, gBK_night, gBK_day, gBK_night]
        G.gNa = [150*nS]*2 + [229*nS]*2
        G.ENa = [ENa_day]*2 + [ENa_night]*2
        G.EK = [EK_day]*2 + [EK_night]*2

        self['labels'] = ['day','BK','ATPase','ATPase+BK']

        mon = StateMonitor(G, ['Inoise'], record=[0,2], name='mon')
        vmon = StateMonitor(G, ['v'], record=True, name='vmon')
        spk = SpikeMonitor(G, record=True, name='spikes')

        @network_operation(dt=10*second)
        def progress(t):
            self.out(f'T = {t/second:.1f} seconds')

        net = Network(G, mon, vmon, spk, Noise, progress, name='clock_network')
        net.run(dur)

        self.save_simulation('sim', net)

    @step
    def plot(self, figsize=(6.52380952, 7.60544218)):
        """Plot figure of diurnal simulation."""
        S = self.read_simulation('sim')
        sdf = S.mon
        vdf = S.vmon
        spikes = S.spikes
        tlim = (self.c.warmup, self.c.warmup + self.c.dur)

        f = self.figure('activity', clear=True, figsize=figsize,
                title='DN1p Clock Neuron Simulation')
        axs = f.add_subplot(311)
        axv = f.add_subplot(312, sharex=axs)
        axn = f.add_subplot(313, sharex=axs)

        labels = self.c.labels
        N_cells = len(labels)
        cols = 'bygr'

        def plot_spikes(ax):
            for i in range(N_cells):
                st = spikes.loc[spikes.neuron == i, 't']
                nspikes = len(st)
                ax.plot(st, [N_cells-i]*nspikes, c=cols[i], ls='', marker='o',
                        mfc='none', alpha=0.6)
            ax.set_ylabel('Spikes')
            ax.set_yticks(list(range(1,N_cells+1)))
            ax.set_yticklabels(labels[::-1])
        plot_spikes(axs)

        def plot_voltage(ax):
            for i in range(N_cells):
                trace = vdf.loc[vdf.neuron == i]
                ax.plot(trace.t, 1000*trace.v, lw=0.8, alpha=0.6, c=cols[i],
                        label=labels[i])
            ax.set_ylabel(r'$V_m$ (mV)')
            ax.legend(loc='upper left')
        plot_voltage(axv)

        def plot_noise(ax):
            for i,c,label in [(0,'b','day'),(2,'g','night')]:
                trace = sdf.loc[sdf.neuron == i]
                ax.plot(trace.t, 1e12*trace.Inoise, c=c, lw=0.6, alpha=0.7)
            ax.set_ylabel('Slow noise (pA)')
            ax.set_xlabel('Simulation time (s)')
        plot_noise(axn)

        # Truncate the warmup period
        axs.set_xlim(*tlim)

    @step
    def plot_figure(self, h=0.04, figsize=(6.52380952,7.60544218)):
        """Figure of model results for paper."""
        S = self.read_simulation('sim')
        sdf = S.mon
        vdf = S.vmon
        spikes = S.spikes

        f = self.figure('figure', clear=True, figsize=figsize,
                title='Spike Shape and Irregularity of Model Activity')
        axs = f.add_subplot(211)
        axi = f.add_subplot(212)

        labels = self.c.labels
        N_cells = len(labels)
        cols = 'bygr'

        def plot_spikeshape(ax):
            t0 = np.linspace(0.0, N_cells*2*h, N_cells)
            for i in range(N_cells):
                st = spikes.loc[spikes.neuron == i, 't']
                st = st[st>self.c.warmup]
                vtrace = vdf.loc[vdf.neuron == i]
                dt, tr = spike_traces(st, vtrace.t, vtrace.v, half_window=h)
                if dt is None:
                    continue
                ax.plot(dt + t0[i], 1000.0*tr, c='0.5', ls='-', lw=0.5,
                        alpha=0.1, zorder=0)
                ax.plot(dt + t0[i], 1000.0*tr.mean(axis=1), c=cols[i], ls='-',
                        lw=1.3, alpha=0.9, zorder=1)

            ax.set_xticks(t0)
            ax.set_xticklabels(labels)
            ax.set_ylabel(r'$V_m$ (mV)')
        plot_spikeshape(axs)

        def plot_isi(ax):
            dataset = []
            for i in range(N_cells):
                st = spikes.loc[spikes.neuron == i, 't'].values
                st = st[st>self.c.warmup]
                if len(st) < 2:
                    dataset.append([1])
                    continue
                isi = np.diff(st)
                isi /= np.median(isi)
                isi = np.log10(isi)
                dataset.append(isi)
            vln = ax.violinplot(dataset, widths=0.75, showmedians=True,
                    points=128)
            [body.set_alpha(0.6) for body in vln['bodies']]
            [body.set_color(cols[i]) for i,body in enumerate(vln['bodies'])]
            vln['cmaxes'].set_visible(False)
            vln['cmins'].set_visible(False)
            vln['cmedians'].set_color('k')
            vln['cmedians'].set_alpha(0.6)
            vln['cbars'].set_color('k')
            vln['cbars'].set_alpha(0.6)
            ax.set_xticks(list(range(1,N_cells+1)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Log median-normalized ISI')
            ax.set_xlabel('DN1p Model')
        plot_isi(axi)

    @step
    def detect_spikes(self, trace_window=0.05):
        """Run adaptive spike detection on voltage traces to find spikes."""
        vdf = self.read_simulation('sim').vmon

        for i in np.unique(vdf.neuron):
            trace = vdf.loc[vdf.neuron == i]
            st = find_spikes(trace.t.values, 1000.0*trace.v.values,
                    half_window=0.004, dv_thresh=4.0, wait=0.005, cut_pct=0.9)
            self.save_array(f'spikes/neuron_{i:02d}', st)

            dt, tr = spike_traces(st[st > self.c.warmup], trace.t, trace.v,
                    half_window=trace_window)
            self.save_array(f'spikes/traces_{i:02d}', tr)
        self.save_array('spikes/traces_dt', dt)

    @step
    def plot_main_figure(self, pad=0.1, tracewin=0.05, nspiketraces=200,
        trace_interval=(26,32), tracescale=1.0, nrasters=6, rasterdur=10.0,
        rasterscale=5.0, isilim=(0.5,4), isibins=128, notebook=True,
        figsize=(10.41496599,6.54421769)):
        """Main figure draft for the paper."""
        if notebook:
            mpl.rcParams['font.size'] = 8.0
            mpl.rcParams['axes.labelpad'] = 1.7
            mpl.rcParams['xtick.major.pad'] = 1.7
            mpl.rcParams['ytick.major.pad'] = 1.7
            mpl.rcParams['figure.subplot.wspace'] = 0.29
            mpl.rcParams['legend.frameon'] = False

        S = self.read_simulation('sim')
        tmin, tmax = trace_interval

        # Set up noise trace: choose the control neuron's noise signal
        sdf = S.mon.loc[S.mon.neuron == 0, ['t', 'Inoise']].copy()
        sdf.rename(columns={'Inoise': 'I'}, inplace=True)
        sdf.I *= 1e12  # convert to pA

        # Set up voltage traces
        vdf = S.vmon.copy()
        vdf.v *= 1e3  # convert to mV

        # Labels and colors
        labels = self.c.labels
        N_cells = len(labels)
        nx = np.arange(N_cells)
        cols = mpl.cm.Dark2(np.linspace(0,1,N_cells))

        f = self.figure('paper', clear=True, figsize=figsize,
                title=f'Draft Figure for ATPase+BK Model: {self.c.dur:.1f}-s '
                       'Simulation')
        gshape = (2,4)
        ax_trace = plt.subplot2grid(gshape, (0,0))
        ax_spikes = plt.subplot2grid(gshape, (0,1), colspan=2)
        ax_stats = plt.subplot2grid(gshape, (0,3))
        ax_raster = plt.subplot2grid(gshape, (1,0))
        ax_isi = plt.subplot2grid(gshape, (1,1), colspan=2)
        ax_cv = plt.subplot2grid(gshape, (1,3))

        # Plot formatting parameters
        noisekw = dict(c='k', ls='-', lw=0.6, alpha=0.7)
        noisezerokw = dict(c='0.5', lw=0.4, zorder=-1)
        scalekw = dict(c='k', ls='-', lw=1.8)#, solid_capstyle='butt')
        exspkkw = dict(marker='|', ls='', ms=9, mew=0.5, mfc='none')
        rasterkw = dict(marker='|', ls='', ms=5, mew=0.4, mfc='none')
        vkw = dict(ls='-', lw=0.5, alpha=0.8)
        spktrkw = dict(ls='-', c='0.5', lw=0.5, alpha=0.12, zorder=0)
        spkmeankw = dict(ls='-', lw=1.3, alpha=0.9, solid_capstyle='round',
                zorder=1)
        acol = '#3343c0'
        scol = '#444444'
        slopekw = dict(ls='-', lw=0.8, c=scol, marker='s', ms=7, mfc=scol, mec='w', mew=1.5)
        ahpkw = dict(ls='-', lw=0.8, c=acol, marker='o', ms=7, mfc=acol, mec='w', mew=1.5)
        labelkw = dict(fontsize='small', ha='right', va='center')

        def _norm(x, b, a):
            x = np.asarray(x)
            return (a-b) * (x-x.min())/x.ptp() + b

        def plot_trace(ax):
            noise = sdf.loc[(sdf.t >= tmin) & (sdf.t <= tmax)]
            trace = vdf.loc[(vdf.t >= tmin) & (vdf.t <= tmax)]
            ax.plot(noise.t, _norm(noise.I, 0.7, 1.0), **noisekw)
            zero = (-noise.I.min()/noise.I.ptp()) * 0.3 + 0.7
            ax.plot([tmin, tmax], [zero]*2, **noisezerokw)
            ax.text(tmin-0.1, zero, '0', **labelkw)

            # Time/Noise-current right-angle scale bar
            scalex = tmax - 0.1
            scaley = 0.99
            noiselen = 2.0
            ax.plot([scalex-tracescale, scalex], [scaley]*2, **scalekw)
            ax.plot([scalex, scalex],
                    [scaley-(noiselen/noise.I.ptp())*(1.0-0.7), scaley],
                    **scalekw)
            self.out(f'Trace scale bar = {tracescale} seconds, {noiselen} pA')

            y_spk = np.linspace(0.65, 0.45, N_cells)
            v_traces = []
            v_t = None
            for i in range(N_cells):
                st = self.read_array(f'spikes/neuron_{i:02d}')
                st = st[(st >= tmin) & (st <= tmax)]
                nspikes = st.size
                ax.plot(st, [y_spk[i]]*nspikes, c=cols[i], **exspkkw)
                ax.text(tmin-0.1, y_spk[i], labels[i], **labelkw)

                # Collect the voltage traces for norm'n before plotting
                trace = vdf.loc[(vdf.t >= tmin) & (vdf.t <= tmax) &
                                (vdf.neuron == i)]
                v_traces.append(trace.v.values)
                if v_t is None:
                    v_t = trace.t.values

            v_traces = _norm(v_traces, 0, 0.4)
            for i in reversed(range(N_cells)):
                ax.plot(v_t, v_traces[i], c=cols[i], **vkw)
            ax.text(tmin-0.1, 0.2, r'$V_m$', **labelkw)

            ax.set_ylim(0, 1)
            ax.set_xlim(tmin, tmax)
            ax.set_axis_off()
        plot_trace(ax_trace)

        mean_spikes = []
        def plot_spikes(ax):
            t0 = np.linspace(0.0, (1+pad)*(N_cells-1)*2*tracewin, N_cells)
            self.out(f'Showing up to {nspiketraces} example spike traces.')
            dt = self.read_array('spikes/traces_dt')
            ix = np.abs(dt) <= tracewin
            for i in range(N_cells):
                tr = 1000.0*self.read_array(f'spikes/traces_{i:02d}')  # mV
                if not tr.size:
                    continue
                ax.plot(dt[ix] + t0[i], tr[ix,:nspiketraces], **spktrkw)
                tr_mean = tr.mean(axis=1)
                mean_spikes.append(tr_mean)
                ax.plot(dt[ix] + t0[i], tr_mean[ix], c=cols[i], label=labels[i],
                        **spkmeankw)

            ax.set_xticks(t0)
            ax.set_xticklabels(labels)
            ax.set_ylabel(r'$V_m$ (mV)')
            ra.despine(ax)
        plot_spikes(ax_spikes)

        def plot_stats(ax):
            maxslope = np.zeros(N_cells)
            ahp = np.zeros(N_cells)
            for i in range(N_cells):
                maxslope[i] = np.max(np.diff(mean_spikes[i])) / 0.1  # mV/ms
                ahp[i] = mean_spikes[i].min() - mean_spikes[i][0]  # mV

            ax.plot(nx, ahp, label='AHP', **ahpkw)
            ax2 = ax.twinx()
            ax2.plot(nx, maxslope, label='Slope', **slopekw)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

            ax.set_ylim(-25, 0)
            ax2.set_ylim(0, 100)
            ax.set_ylabel('AHP (mV)')
            ax2.set_ylabel('Maximum A.P. slope (mV/ms)', rotation=270,
                    labelpad=9.5)
            ax.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax.set_xticks(nx)
            ax.set_xticklabels(labels)
            ax2.legend(h1 + h2, l1 + l2, loc='upper left')
        plot_stats(ax_stats)

        def plot_rasters(ax):
            totrasters = N_cells * nrasters
            tp = np.linspace(self.c.warmup, self.c.warmup + self.c.dur -
                    rasterdur, nrasters)
            yp = np.linspace(1, 0, (N_cells+1) * nrasters - 1)

            k = 0
            for i in range(N_cells):
                spk = self.read_array('spikes/neuron_%02d' % i)
                ax.text(-0.2, yp[k+int(nrasters/2)-1], labels[i],
                        fontsize='small', ha='right', va='top')
                for j in range(nrasters):
                    st = spk[(spk >= tp[j]) & (spk <= tp[j] + rasterdur)]
                    st -= tp[j]
                    ax.plot(st, [yp[k]]*st.size, c=cols[i], **rasterkw)
                    k += 1
                k += 1

            ax.plot([rasterdur-rasterscale, rasterdur], [0.025]*2, **scalekw)
            self.out(f'Raster scale bar = {rasterscale} seconds')
            ax.set_xlim(0, rasterdur)
            ax.set_axis_off()
        plot_rasters(ax_raster)

        def plot_isi_distros(ax):
            edges = np.logspace(np.log10(isilim[0]), np.log10(isilim[1]),
                    isibins+1)
            width = 1 - pad
            for i in range(N_cells):
                spk = self.read_array('spikes/neuron_%02d' % i)
                spk = spk[spk > self.c.warmup]
                isi = np.diff(spk)
                histo, _ = np.histogram(isi/np.median(isi), bins=edges)

                ax.axvline(i, c='0.5', lw=0.5, zorder=1)
                ax.bar( i, np.diff(edges),
                        width=-width * histo/histo.max(),
                        bottom=edges[:-1], color=cols[i], edgecolor='k',
                        linewidth=0.0, orientation='horizontal', align='edge')

            logticks = map(lambda y: round(y, 1),
                    np.logspace(np.log10(isilim[0]), np.log10(isilim[1]), 5))
            for y in logticks:
                ax.axhline(y, c='0.5', lw=0.5, zorder=-1)
                ax.text(-1-1.2*pad, y, f'{y:.1f}', **labelkw)
            ax.axhline(1.0, c='k', ls='--', lw=0.5, zorder=2)
            ax.text(-1-1.2*pad, 1.0, '1.0', **labelkw)

            ax.set_yscale('log')
            ax.set_ylim(edges[0], edges[-1])
            ax.set_xlim(-1-pad, N_cells-1+pad)
            ax.yaxis.set_visible(False)
            ax.set_xticks(np.arange(N_cells) - 0.5)
            ax.set_xticklabels(labels)
            ra.despine(ax, left=False)
            ra.ylabel(ax, 'Log median-normalized ISI')
        plot_isi_distros(ax_isi)

        def plot_cv(ax):
            meanrate = np.zeros(N_cells)
            cv = np.zeros(N_cells)
            for i in range(N_cells):
                spk = self.read_array('spikes/neuron_%02d' % i)
                spk = spk[spk > self.c.warmup]
                isi = np.diff(spk)
                meanrate[i] = spk.size / self.c.dur  # spikes/s
                cv[i] = isi.std() / isi.mean()
                # cv[i] = np.sqrt(np.exp(np.log(isi).std()**2) - 1)  # geometric CV

            ax.plot(nx, cv, label='CV', **ahpkw)
            ax2 = ax.twinx()
            ax2.plot(nx, meanrate, label='Rate', **slopekw)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

            ax.set_ylabel('Spike irregularity (CV)')
            ax2.set_ylabel('Average firing rate (spikes/s)', rotation=270,
                    labelpad=9.5)
            ax.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax.set_ylim(0, 1)
            ax2.set_ylim(0, 10)
            ax.set_xticks(nx)
            ax.set_xticklabels(labels)
            ax2.legend(h1 + h2, l1 + l2, loc='lower left')
        plot_cv(ax_cv)
