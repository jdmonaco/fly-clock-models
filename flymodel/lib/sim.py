"""
Simulation support functions.
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from brian2 import *


def triangle_wave(t0, p):
    """Triangle wave on [0,1] with period p for time points t0."""
    t = t0 - p/4
    a = p/2
    b = np.floor(t/a + 1/2)
    return 0.5 + (1/a) * (t - a*b) * (-1)**b

def OU_process(x0, mu, sigma, theta, duration, dt):
    """An Ornstein-Uhlenbeck process with arbitrary dt.

    This is taken from a SO answer here: https://stackoverflow.com/a/43120245
    and should be built upon and tested.
    """
    x_t = x0
    # ou(t + dt) = ou(t) + (mu - ou(t)) * (1 - np.exp(-theta*dt)) \
        # + st.norm(loc=0,
                  # scale=sigma*np.sqrt(1/(2*theta)*(1 - np.exp(-2*theta*dt)))).rvs()

#
# Python example from Wikipedia:
# https://en.wikipedia.org/wiki/Euler–Maruyama_method#Computer_implementation
#

def OU_brian_test():
    """Testing simulating an OU process as a Brian stochastic."""
    defaultclock.dt = 0.1 * ms

    N_trains = 5
    duration = 0.1 * second
    x_init = 0.0

    mu = 1.5
    sigma = np.sqrt(0.1) * 0.06 / sqrt(ms)
    theta = 30.0 / second

    eqn = """
          dx/dt = theta * (mu - x) + sigma * xi : 1
          """

    G = NeuronGroup(N_trains, model=eqn, method='euler')
    G.x = x_init

    mon = StateMonitor(G, 'x', record=True)
    run(duration)

    print('Plotting....')
    plt.ioff()
    f = plt.figure(num='OU-brian-test')
    f.clear()
    f.suptitle('OU Test')

    ax = f.add_subplot(111)
    for i in range(N_trains):
        ax.plot(mon.t/ms, mon.x[i])

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('OU process')

    plt.show()
    plt.ion()

def OU_process_wiki(num_sims=5, t_init=3, t_end=7, y_init=0, N=1000,
    c_theta=0.7, c_mu=1.5, c_sigma=0.06):
    """Ornstein-Uhlenbeck process with Euler-Murayama method integration."""
    plt.ioff()
    plt.figure()

    def mu(y, t):
        """Implement the Ornstein–Uhlenbeck mu.""" ### = \theta \cdot (\mu-Y_t)
        return c_theta * (c_mu - y)

    def sigma(y, t):
        """Implement the Ornstein–Uhlenbeck sigma.""" ### = \sigma
        return c_sigma

    def dW(delta_t):
        """Sample a random number at each call."""
        return np.random.normal(loc = 0.0, scale = np.sqrt(delta_t))

    dt    = float(t_end - t_init) / N
    ts    = np.arange(t_init, t_end, dt)
    ys    = np.zeros(N)

    ys[0] = y_init

    for i_sim in range(num_sims):
        for i in range(1, ts.size):
            t_ = (i-1) * dt
            y_ = ys[i-1]
            ys[i] = y_ + mu(y_, t_) * dt + sigma(y_, t_) * dW(dt)
        plt.plot(ts, ys)

    plt.show()
