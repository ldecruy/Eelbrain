# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import izip
from math import floor, log10
import os

import matplotlib as mpl
import numpy as np

from . import plot, testnd
from .plot._base import POINT
from ._data_obj import combine


# usage:  with mpl.rc_context(RC):
RC = {'figure.dpi': 300,
      'savefig.dpi': 300,
      'font.family': 'sans-serif',
      'font.sans-serif': 'Helvetica',
      'font.size': 9,
      'figure.subplot.top': 0.975,
      'legend.fontsize': 6,
      'legend.frameon': False}
for key in mpl.rcParams:
    if 'width' in key:
        RC[key] = mpl.rcParams[key] * 0.5


def cname(cid):
    if isinstance(cid, tuple):
        return '-'.join(map(str, cid))
    else:
        return str(cid)


class ClusterPlotter(object):
    """Make plots for spatio-temporal clusters (returned by :meth:`MneExperiment.load_result_plotter`)

    Parameters
    ----------
    ds : Dataset
        Dataset with the data on which the test is based.
    res : Result
        Test result object with spatio-temporal cluster test result.

    rc : dict
        Matplotlib rc-parameters dictionary (the default is optimized for the
        default plot size)
    font_size : int
        Font size for plots (default is 9).
    """
    def __init__(self, ds, res, colors, dst, vec_fmt='pdf', pix_fmt='png',
                 labels=None, h=1.1, rc=None, font_size=None):
        if rc is None:
            rc = RC.copy()
        if font_size is not None:
            rc['font.size'] = font_size
        self.ds = ds
        self.res = res
        self.colors = colors
        self.labels = labels
        self.h = h
        self.rc = rc
        self._dst_vec = os.path.join(dst, '%%s.%s' % vec_fmt)
        self._dst_pix = os.path.join(dst, '%%s.%s' % pix_fmt)
        self._is_anova = isinstance(self.res, testnd.anova)
        if not os.path.exists(dst):
            os.mkdir(dst)

    def _ids(self, ids):
        if isinstance(ids, (float, int)):
            return self._ids_for_p(ids)
        elif isinstance(ids, dict):
            if not self._is_anova:
                raise TypeError("ids can not be dict for results other than ANOVA")
            out = []
            for effect, cids in ids.iteritems():
                if isinstance(cids, float):
                    out.extend(self._ids_for_p(cids, effect))
                else:
                    out.extend((effect, cid) for cid in cids)
            return out
        else:
            return ids

    def _ids_for_p(self, p, effect=None):
        "Find cluster IDs for clusters with p-value <= p"
        idx = self.res.clusters['p'] <= p
        if effect is not None:
            idx = np.logical_and(idx, self.res.clusters[:, 'effect'] == effect)
        ids = self.res.clusters[idx, 'id']
        if self._is_anova:
            effect = self.res.clusters[idx, 'effect']
            return zip(effect, ids)
        else:
            return ids

    def _get_clusters(self, ids):
        return [self._get_cluster(cid) for cid in ids]

    def _get_cluster(self, cid):
        if self._is_anova:
            effect, cid = cid
            return self.res.cluster(cid, effect)
        else:
            return self.res.cluster(cid)

    def plot_color_list(self, name, cells, w=None, colors=None):
        if colors is None:
            colors = self.colors

        with mpl.rc_context(self.rc):
            p = plot.ColorList(colors, cells, self.labels, w=w, show=False)
            p.save(self._dst_vec % "colorlist %s" % name, transparent=True)
            p.close()

    def plot_color_grid(self, name, row_cells, column_cells):
        with mpl.rc_context(self.rc):
            p = plot.ColorGrid(row_cells, column_cells, self.colors, labels=self.labels)
            p.save(self._dst_vec % "colorgrid %s" % name, transparent=True)
            p.close()

    def plot_clusters_spatial(self, ids, views, w=600, h=480, prefix=''):
        """Plot spatial extent of the clusters

        Parameters
        ----------
        ids : sequence | dict | scalar <= 1
            IDs of the clusters that should be plotted. For ANOVA results, this
            should be an ``{effect_name: id_list}`` dict. Instead of a list of
            IDs a scalar can be provided to plot all clusters with p-values
            smaller than this.
        views : str | list of str | dict
            Can a str or list of str to use the same views for all clusters. A dict
            can have as keys labels or cluster IDs.
        w, h : int
            Size in pixels. The default (600 x 480) corresponds to 2 x 1.6 in
            at 300 dpi.
        prefix : str
            Prefix to use for the image files (optional, can be used to
            distinguish different groups of images sharing the same color-bars).

        Notes
        -----
        The horizontal colorbar is 1.5 in wide, the vertical colorbar is 1.6 in
        high.
        """
        ids = self._ids(ids)
        clusters = self._get_clusters(ids)
        clusters_spatial = [c.sum('time') for c in clusters]
        if isinstance(views, basestring):
            views = (views,)

        # vmax
        vmin = min(c.min() for c in clusters_spatial)
        vmax = max(c.max() for c in clusters_spatial)
        abs_vmax = max(vmax, abs(vmin))

        # anatomical extent
        brain_colorbar_done = False
        for cid, cluster in izip(ids, clusters_spatial):
            name = cname(cid)
            if prefix:
                name = prefix + ' ' + name

            for hemi in ('lh', 'rh'):
                if not cluster.sub(source=hemi).any():
                    continue
                brain = plot.brain.cluster(cluster, abs_vmax, views='lat',
                                           background=(1, 1, 1), colorbar=False,
                                           parallel=True, hemi=hemi, w=w, h=h)
                for view in views:
                    brain.show_view(view)
                    brain.screenshot('rgba', True)
                    brain.save_image(self._dst_pix % ' '.join((name, hemi, view)))

                if not brain_colorbar_done:
                    with mpl.rc_context(self.rc):
                        label = "Sum of %s-values" % cluster.info['meas']
                        clipmin = 0 if vmin == 0 else None
                        clipmax = 0 if vmax == 0 else None
                        if prefix:
                            cbar_name = '%s cbar %%s' % prefix
                        else:
                            cbar_name = 'cbar %s'

                        h_cmap = 0.7 + POINT * mpl.rcParams['font.size']
                        p = brain.plot_colorbar(label, clipmin=clipmin, clipmax=clipmax,
                                                width=0.1, h=h_cmap, w=1.5, show=False)
                        p.save(self._dst_vec % cbar_name % 'h', transparent=True)
                        p.close()

                        w_cmap = 0.8 + 0.1 * abs(floor(log10(vmax)))
                        p = brain.plot_colorbar(label, clipmin=clipmin, clipmax=clipmax,
                                                width=0.1, h=1.6, w=w_cmap,
                                                orientation='vertical', show=False)
                        p.save(self._dst_vec % cbar_name % 'v', transparent=True)
                        p.close()

                        brain_colorbar_done = True

                brain.close()

    def _get_data(self, model, sub, subagg):
        """Plot values in cluster

        Parameters
        ----------
        subagg : str
           Index in ds: within index, collapse across other predictors.
        """
        ds = self.ds
        modelname = model

        if sub:
            ds = ds.sub(sub)
            modelname += '[%s]' % sub

        if subagg:
            idx_subagg = ds.eval(subagg)
            ds_full = ds.sub(np.invert(idx_subagg))
            ds_agg = ds.sub(idx_subagg).aggregate("subject", drop_bad=True)
            ds = combine((ds_full, ds_agg), incomplete='fill in')
            ds['condition'] = ds.eval(model).as_factor()
            model = 'condition'
            modelname += '(agg %s)' % subagg

        return ds, model, modelname

    def plot_values(self, ids, model, ymax, ymin=0, dpi=300, sub=None,
                    subagg=None, cells=None, pairwise=False, colors=None):
        """Plot values in cluster

        Parameters
        ----------
        ids : sequence | dict | scalar <= 1
            IDs of the clusters that should be plotted. For ANOVA results, this
            should be an ``{effect_name: id_list}`` dict. Instead of a list of
            IDs a scalar can be provided to plot all clusters with p-values
            smaller than this.

        subagg : str
           Index in ds: within index, collapse across other predictors.
        cells : sequence of cells in model
            Modify visible cells and their order. Only applies to the barplot.
            Does not affect filename.
        pairwise : bool
            Add pairwise tests to barplots.
        colors : dict
            Substitute colors (default are the colors provided at
            initialization).
        """
        ds, model, modelname = self._get_data(model, sub, subagg)
        ids = self._ids(ids)
        if colors is None:
            colors = self.colors

        src = ds['srcm']
        legend_done = False
        n_cells = len(ds.eval(model).cells)
        w_bar = (n_cells * 2 + 4) * (self.h / 12)
        with mpl.rc_context(self.rc):
            for cid in ids:
                name = cname(cid)
                cluster = self._get_cluster(cid)
                y_mean = src.mean(cluster != 0)
                y_tc = src.mean(cluster.any('time'))

                # barplot
                p = plot.Barplot(y_mean, model, 'subject', None, cells, pairwise,
                                 ds=ds, trend=False, corr=None,
                                 title=None, frame=False, yaxis=False, ylabel=False,
                                 colors=colors, bottom=ymin, top=ymax, w=w_bar, h=self.h,
                                 xlabel=None, xticks=None,
                                 tight=False, test_markers=False, show=False)
                p.save(self._dst_vec % ' '.join((name, modelname, 'barplot')), dpi=dpi, transparent=True)
                p.close()

                # time-course
                p = plot.UTSStat(y_tc, model, match='subject', ds=ds, error='sem',
                                 colors=colors, title=None, axtitle=None, frame=False,
                                 bottom=ymin, top=ymax,
                                 legend=None, ylabel=None, xlabel=None, w=self.h * 2, h=self.h,
                                 tight=False, show=False)
                dt = y_tc.time.tstep / 2.
                mark_start = cluster.info['tstart'] - dt
                mark_stop = cluster.info['tstop'] - dt
                p.add_vspan(mark_start, mark_stop, color='k', alpha=0.1, zorder=-2)
                p.save(self._dst_vec % ' '.join((name, modelname, 'timecourse')), dpi=dpi, transparent=True)
                p.close()

                # legend
                if not legend_done:
                    p.save_legend(self._dst_vec % (modelname + ' legend'), transparent=True)
                    legend_done = True