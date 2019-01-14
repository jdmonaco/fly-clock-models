"""
Functions for plotting and saving 2D maps and colorbars/discs.
"""

from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
import numpy as np

from pouty import debug
from roto import radians, circstats
from roto.lines import cmapline
from roto.images import rgba_to_image, map_to_rgba
from roto.colormaps import default_cmap

from ..lib.lfp import PHASE_MIN, PHASE_MAX
from ..lib.motion import arena_extent


def _handle_levels(M, kwargs):
    """Discretize colormaps into levels if specified in map call keywords.

    The `kwargs` dict is updated with color limit defaults and a new
    discretized colormap if `levels` is specified.

    Returns (cmap, norm) tuple.
    """
    levels = kwargs.pop('levels', None)
    circular = kwargs.pop('circular', False)
    has_satmap = M.ndim == 3 and M.shape[0] == 2

    if has_satmap or (circular and np.nanmax(np.abs(M)) < PHASE_MAX):
        dftmin, dftmax = PHASE_MIN, PHASE_MAX
    else:
        dftmin, dftmax = np.nanmin(M), np.nanmax(M)
    cmin = kwargs.get('cmin', dftmin)
    cmax = kwargs.get('cmax', dftmax)
    debug('handle_levels: setting cmin={:.2f}, cmax={:.2f}', cmin, cmax)

    norm = None
    cmap = kwargs.get('cmap', default_cmap(circular=has_satmap or circular))
    kwargs.update(cmin=cmin, cmax=cmax, cmap=cmap)

    if levels is not None:
        if type(levels) is int:
            debug('handle_levels: computing {} color levels', levels)
            cvals = np.linspace(0, 1, levels)
            if has_satmap or circular == True:
                cvals *= (levels - 1) / levels
            colors = plt.get_cmap(cmap)(cvals)
        else:
            colors = levels
            levels = len(colors)
        boundaries = np.linspace(cmin, cmax, levels + 1)
        cmap, norm = from_levels_and_colors(boundaries, colors)
        kwargs.update(cmap=cmap)

    return cmap, norm

def trace(x, y, values, ax=None, extent=None, cbar=False, cbargs={}, lcargs={},
    **kwargs):
    """Plot a spatial line trace of a mapping function.

    Similar arguments as `plot` except (x, y, values) are used as the
    arguments to `roto.lines.cmapline` instead of the matrix image argument.
    LineCollection keywords may be specified via `lcargs`.
    """
    cmap, norm = _handle_levels(values, kwargs)
    extent = extent or arena_extent('circle')

    ax = ax or plt.gca()
    lcargs.update(cmap=cmap, norm=norm)
    lc = cmapline(x, y, values[:x.size-1], ax=ax, **lcargs)

    ax.axis(extent)
    ax.axis('equal')

    if cbar:
        cb = plt.colorbar(lc, **cbargs)
        return lc, cb
    return lc

def plot(M, ax=None, extent=None, cbar=False, cbargs={}, **kwargs):
    """Plot a map image to an axis.

    Remaining keyword arguments are passed to `tools.images.array_to_rgba`.

    Arguments:
    M -- an (m,n) or (2,m,n)-shaped map matrix array

    Keyword arguments:
    ax -- existing axis to plot map
    extent -- scalars (left, right, bottom, top) for map data extent
    cbar -- add a colorbar to the axis (requires adding an invisible image)
    cbargs -- keyword arguments to send to `Figure.colorbar(...)`

    Other kwargs:
    levels -- optional, number of levels to discretize the map colors
    circular -- whether a 2D map matrix has circular data

    Returns:
    image, colorbar -- if cbar == True
    image -- otherwise
    """
    cmap, norm = _handle_levels(M, kwargs)
    extent = extent or arena_extent('circle')
    imargs = dict(interpolation='nearest', origin='lower',
                  cmap=cmap, norm=norm, extent=extent, aspect='equal')

    ax = ax or plt.gca()
    if cbar:
        f = ax.get_figure()
        mat = M
        if mat.ndim == 3:
            mat = M[0]
        cbimargs = imargs.copy()
        if norm is None:
            cbimargs.update(vmin=kwargs['cmin'], vmax=kwargs['cmax'])
        cbim = ax.imshow(mat, **cbimargs)
        cb = f.colorbar(cbim, ax=ax, **cbargs)
        cbim.set_visible(False)

    im = ax.imshow(map_to_rgba(M, **kwargs), **imargs)
    plt.draw_if_interactive()

    if cbar:
        return im, cb
    return im

def save(M, filename, **kwargs):
    """Save a map matrix to an image file.

    Keyword arguments are passed to `images.array_to_rgba` except for `levels`
    and `circular` which behave as in `plot(...)`.
    """
    _handle_levels(M, kwargs)
    rgba_to_image(map_to_rgba(M, **kwargs), filename)

def _fmtdeg(d):
    if np.isclose(d, 360):
        return 360
    return '%g' % (d % 360)

def phasebar(ax=None, degrees=False, aspect=12.0, cycles=2, bins=64,
    levels=None, labelsize='medium'):
    """Plot a phase/coherence 2D colorbar.

    Keyword arguments:
    ax -- existing axis to plot the colorbar
    degrees -- display tick labels in degrees (default: radians)
    aspect -- aspect ratio of the 2D colorbar
    cycles -- number of phase cycles to display
    bins -- resolution of the colorbar in pixel rows
    levels -- contour-like discrete color levels
    """
    ax = ax or plt.gca()
    cycles = int(cycles)
    pmax = degrees and 180.0 or np.pi
    phase = np.tile(np.linspace(-pmax, pmax, bins), cycles)
    ptot = 2 * pmax * cycles
    length = np.linspace(0, 1, bins)
    pcmat = np.empty((2, phase.size, bins))
    pcmat[0] = np.atleast_2d(phase).T
    pcmat[1] = length
    im = plot(pcmat, ax=ax, extent=[0,aspect,0,1], levels=levels)
    ax.set_yticks([0, 1])
    pticks = np.linspace(0, aspect, 2 * cycles + 1)
    pticklabels = np.linspace(0, ptot, 2 * cycles + 1)
    ax.set_xticks(pticks)
    if degrees:
        ax.set_xticklabels([_fmtdeg(p) for p in pticklabels], size=labelsize)
    else:
        ax.set_xticklabels([radians.prettify(p, modtwopi=True)
            for p in pticklabels], size=labelsize)
    ax.xaxis.tick_top()
    plt.draw_if_interactive()
    return im

def phasedisc(ax=None, degrees=False, res=64, levels=None, orient=None,
    labelpad=0.04, labelsize='medium', inline=False, zoom=0.0):
    """Plot a phase/coherence 2D colordisc.

    Keyword arguments:
    ax -- existing axis to plot the colorbar
    degrees -- display tick labels in degrees (default: radians)
    res -- resolution of the colorbar in pixel rows
    levels -- contour-like discrete color levels
    orient -- specify 'radians' (default) or 'natural'
    labelpad -- fractional padding between ticks and labels
    labelsize -- font size for the tick labels
    inline -- inline radian fractions for radian tick labels
    zoom -- zoom-out factor (zoom=1 makes disc appear half the size)
    """
    ax = ax or plt.gca()
    ax.clear()
    ax.set_axis_bgcolor((0,) * 4)
    side = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(side, side)
    angle = np.arctan2(Y, X)
    length = np.hypot(X, Y)
    if orient == 'natural':
        phase = (angle.T - np.pi / 2) % (2 * np.pi)
        phase[phase > np.pi] -= 2 * np.pi
        phase[:] = np.flipud(phase)
    else:
        phase = (angle - np.pi / 2) % (2 * np.pi)
        phase[phase > np.pi] -= 2 * np.pi
        phase[:] = np.fliplr(phase)
    pcmat = np.array((phase, length))
    pcmat[:, length>1] = np.nan  # mask values outside the unit disc

    pmax = degrees and 360.0 or 2 * np.pi
    pticks = np.array([0, 0.25, 0.5, 0.75]) * pmax
    if degrees:
        pstr = [_fmtdeg(p) for p in pticks]
    else:
        pstr = [radians.prettify(p, inline=inline, modtwopi=True) for p in pticks]
    if orient == 'natural':
        porder = ('top', 'right', 'bottom', 'left')
    else:
        porder = ('right', 'top', 'left', 'bottom')
    plabels = { k:v for k,v in zip(porder, pstr) }

    L = 1 + labelpad
    ax.text(-L, 0, plabels['left'], ha='right', va='center', size=labelsize)
    ax.text(0, L, plabels['top'], ha='center', va='baseline', size=labelsize)
    ax.text(L, 0, plabels['right'], ha='left', va='center', size=labelsize)
    ax.text(0, -L, plabels['bottom'], ha='center', va='top', size=labelsize)

    im = plot(pcmat, ax=ax, extent=[-1,1,-1,1], levels=levels,
            mask_color=(0,) * 4)

    ax.plot([-1,1], [0,0], 'w-', lw=0.8, alpha=0.4)
    ax.plot([0,0], [-1,1], 'w-', lw=0.8, alpha=0.4)

    ax.set(yticks=[], xticks=[], yticklabels=[], xticklabels=[],
        xlim=(-1-zoom,1+zoom), ylim=(-1-zoom,1+zoom))
    for sp in ax.spines:
        ax.spines[sp].set_visible(False)

    plt.draw_if_interactive()
    return im
