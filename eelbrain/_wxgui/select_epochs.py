"""GUI for rejecting epochs"""

# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

# Document:  represents data
# ChangeAction:  modifies Document
# Model:  creates ChangeActions and applies them to the History
# Frame:
#  - visualizaes Document
#  - listens to Document changes
#  - issues commands to Model

from __future__ import division

from copy import deepcopy
from logging import getLogger
import math
import os
import re
import time

import mne
import numpy as np
import wx

from .. import load, save, plot, fmtxt
from .._data_obj import Dataset, Factor, Var, Datalist, corr, asndvar, combine
from .. import _meeg as meeg
from .. import _report
from .._names import INTERPOLATE_CHANNELS
from .._info import BAD_CHANNELS
from .._utils.parse import FLOAT_PATTERN, POS_FLOAT_PATTERN
from ..plot._base import find_axis_params_data, find_axis_params_dim, \
    find_fig_vlims
from ..plot._nuts import _plt_bin_nuts
from ..plot._topo import _ax_topomap
from ..plot._utsnd import _ax_bfly_epoch
from .._utils.numpy_utils import full_slice
from .._wxutils import Icon, ID, REValidator
from .app import get_app
from .frame import EelbrainDialog
from .mpl_canvas import FigureCanvasPanel
from .history import Action, FileDocument, FileModel, FileFrame
from .text import TextFrame, HTMLFrame


# IDs
MEAN_PLOT = -1
TOPO_PLOT = -2
OUT_OF_RANGE = -3


def _epoch_list_to_ranges(elist):
    out = []
    i = 0
    i_last = len(elist) - 1
    start = None
    while i <= i_last:
        cur = elist[i]
        if i < i_last and elist[i + 1] - cur == 1:
            if start is None:
                start = cur
        elif start is None:
            out.append(fmtxt.Link(cur, 'epoch:%i' % cur))
        else:
            out.append(fmtxt.Link(start, 'epoch:%i' % start) + '-' +
                       fmtxt.Link(cur, 'epoch:%i' % cur))
            start = None
        i += 1
    return out


def format_epoch_list(l, head="Epochs by channel:"):
    d = meeg.channel_listlist_to_dict(l)
    if not d:
        return "None."
    out = fmtxt.List(head)
    for ch in sorted(d, key=lambda x: -len(d[x])):
        item = fmtxt.FMText("%s (%i): " % (ch, len(d[ch])))
        item += fmtxt.delim_list(_epoch_list_to_ranges(d[ch]))
        out.add_item(item)
    return out


class ChangeAction(Action):
    """Action objects are kept in the history and can do and undo themselves"""

    def __init__(self, desc, index=None, old_accept=None, new_accept=None,
                 old_tag=None, new_tag=None, old_path=None, new_path=None,
                 old_bad_chs=None, new_bad_chs=None, old_interpolate=None,
                 new_interpolate=None):
        """
        Parameters
        ----------
        desc : str
            Description of the action
            list of (i, name, old, new) tuples
        """
        self.desc = desc
        self.index = index
        self.old_path = old_path
        self.old_accept = old_accept
        self.old_tag = old_tag
        self.new_path = new_path
        self.new_accept = new_accept
        self.new_tag = new_tag
        self.old_bad_chs = old_bad_chs
        self.new_bad_chs = new_bad_chs
        self.old_interpolate = old_interpolate
        self.new_interpolate = new_interpolate

    def do(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.new_accept, self.new_tag,
                         self.new_interpolate)
        if self.new_path is not None:
            doc.set_path(self.new_path)
        if self.new_bad_chs is not None:
            doc.set_bad_channels(self.new_bad_chs)

    def undo(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.old_accept, self.old_tag,
                         self.old_interpolate)
        if self.new_path is not None and self.old_path is not None:
            doc.set_path(self.old_path)
        if self.new_bad_chs is not None:
            doc.set_bad_channels(self.old_bad_chs)


class Document(FileDocument):
    """Represents data for the current state of the Document

    Data can be accesses through attributes, but should only be changed through
    the set_...() methods.

    Attributes
    ----------
    n_epochs : int
        The number of epochs.
    epochs : NDVar
        The raw epochs.
    accept : Var of bool
        Case status.
    tag : Factor
        Case tag.
    trigger : Var of int
        Case trigger value.
    blink : Datalist | None
        Case eye tracker artifact data.
    """

    def __init__(self, ds, data='meg', accept='accept', blink='blink',
                 tag='rej_tag', trigger='trigger', path=None, bad_chs=None,
                 allow_interpolation=True):
        """
        Parameters
        ----------
        ds : Dataset
            Dataset containing
            ...
        path : None | str
            Default location of the epoch selection file (used for save
            command). If the file exists, it is loaded as initial state.
        """
        if isinstance(ds, mne.Epochs):
            epochs = ds
            if not epochs.preload:
                err = ("Need Epochs with preloaded data (preload=True)")
                raise ValueError(err)
            ds = Dataset()
            ds[data] = epochs
            ds['trigger'] = Var(epochs.events[:, 2])

        data = asndvar(data, ds=ds)
        self.n_epochs = n = len(data)

        if not isinstance(accept, basestring):
            raise TypeError("accept needs to be a string")
        if accept not in ds:
            x = np.ones(n, dtype=bool)
            ds[accept] = Var(x)
        accept = ds[accept]

        if not isinstance(tag, basestring):
            raise TypeError("tag needs to be a string")
        if tag in ds:
            tag = ds[tag]
        else:
            tag = Factor([''], repeat=n, name=tag)
            ds.add(tag)

        if not isinstance(trigger, basestring):
            raise TypeError("trigger needs to be a string")
        if trigger in ds:
            trigger = ds[trigger]
        else:
            err = ("ds does not contain a variable named %r. The trigger "
                   "parameters needs to point to a variable in ds containing "
                   "trigger values." % trigger)
            raise KeyError(err)

        if INTERPOLATE_CHANNELS in ds:
            interpolate = ds[INTERPOLATE_CHANNELS]
            if not allow_interpolation and any(interpolate):
                raise ValueError("Dataset contains channel interpolation "
                                 "information but interpolation is turned off")
        else:
            interpolate = Datalist([[]] * ds.n_cases, INTERPOLATE_CHANNELS,
                                   'strlist')

        if isinstance(blink, basestring):
            if ds is not None:
                blink = ds.get(blink, None)
        elif blink is True:
            if 'edf' in ds.info:
                tmin = data.time.tmin
                tmax = data.time.tmax
                _, blink = load.eyelink.artifact_epochs(ds, tmin, tmax,
                                                        esacc=False)
            else:
                msg = ("No eye tracker data was found in ds.info['edf']. Use "
                       "load.eyelink.add_edf(ds) to add an eye tracker file "
                       "to a Dataset ds.")
                wx.MessageBox(msg, "Eye Tracker Data Not Found")
                blink = None
        elif blink is not None:
            raise TypeError("blink needs to be a string or None")

        if blink is not None:
            raise NotImplementedError("Frame.SetPage() needs to be updated to "
                                      "use blink information")

        # options
        self.allow_interpolation = allow_interpolation

        # data
        self.epochs = data
        self.accept = accept
        self.tag = tag
        self.interpolate = interpolate
        self.trigger = trigger
        self.blink = blink
        self.bad_channels = []
        self.good_channels = None

        # cache
        self._good_sensor_indices = {}

        # publisher
        self._bad_chs_change_subscriptions = []
        self._case_change_subscriptions = []

        # finalize
        if bad_chs is not None:
            self.set_bad_channels_by_name(bad_chs)
        FileDocument.__init__(self, path)

        if path and os.path.exists(path):
            accept, tag, interpolate, bad_chs = self.read_rej_file(path)
            self.accept[:] = accept
            self.tag[:] = tag
            self.interpolate[:] = interpolate
            self.set_bad_channels_by_name(bad_chs)

    @property
    def bad_channel_names(self):
        return [self.epochs.sensor.names[i] for i in self.bad_channels]

    def good_data(self):
        "All cases, only good channels"
        if self.bad_channels:
            return self.epochs.sub(sensor=self.good_channels)
        else:
            return self.epochs

    def good_sensor_index(self, case):
        "Index of non-interpolated sensor relative to good sensors"
        if self.interpolate[case]:
            key = frozenset(self.interpolate[case])
            if key in self._good_sensor_indices:
                return self._good_sensor_indices[key]
            else:
                out = np.ones(len(self.epochs.sensor), bool)
                out[self.epochs.sensor.dimindex(self.interpolate[case])] = False
                if self.good_channels is not None:
                    out = out[self.good_channels]
                self._good_sensor_indices[key] = out
                return out

    def get_epoch(self, case, name):
        if self.bad_channels:
            return self.epochs.sub(case=case, sensor=self.good_channels, name=name)
        else:
            return self.epochs.sub(case=case, name=name)

    def get_grand_average(self):
        "Grand average of all accepted epochs"
        return self.epochs.sub(case=self.accept.x, sensor=self.good_channels,
                               name="Grand Average").mean('case')

    def set_bad_channels(self, indexes):
        """Set the channels to treat as bad (i.e., exclude)

        Parameters
        ----------
        bad_chs : collection of int
            Indices of channels to treat as bad.
        """
        indexes = sorted(indexes)
        if indexes == self.bad_channels:
            return
        self.bad_channels = indexes
        if indexes:
            self.good_channels = np.setdiff1d(np.arange(len(self.epochs.sensor)),
                                              indexes, True)
        else:
            self.good_channels = None
        self._good_sensor_indices.clear()
        for callback in self._bad_chs_change_subscriptions:
            callback()

    def set_bad_channels_by_name(self, names):
        self.set_bad_channels(self.epochs.sensor.dimindex(names))

    def set_case(self, index, state, tag, interpolate):
        if state is not None:
            self.accept[index] = state
        if tag is not None:
            self.tag[index] = tag
        if interpolate is not None:
            self.interpolate[index] = interpolate

        for func in self._case_change_subscriptions:
            func(index)

    def set_path(self, path):
        """Set the path

        Parameters
        ----------
        path : str
            Path under which to save. The extension determines the way file
            (*.pickled -> pickled Dataset; *.txt -> tsv)
        """
        root, ext = os.path.splitext(path)
        if ext == '':
            path = root + '.txt'
        FileDocument.set_path(self, path)

    def read_rej_file(self, path):
        "Read a file making sure it is compatible"
        _, ext = os.path.splitext(path)
        if ext == '.pickled':
            ds = load.unpickle(path)
        elif ext == '.txt':
            ds = load.tsv(path, delimiter='\t')
        else:
            raise ValueError("Unknown file extension for rejections: %r" % ext)

        # check file
        if ds.n_cases > self.n_epochs:
            app = get_app()
            cmd = app.message_box("The File contains more events than the data "
                                  "(%i vs %i). Truncate the file?"
                                  % (ds.n_cases, self.n_epochs),
                                  "Truncate the file?", wx.OK | wx.CANCEL |
                                  wx.CANCEL_DEFAULT | wx.ICON_WARNING)
            if cmd == wx.OK:
                ds = ds[:self.n_epochs]
            else:
                raise RuntimeError("Unequal number of cases")
        elif ds.n_cases < self.n_epochs:
            app = get_app()
            cmd = app.message_box("The rejection file contains fewer epochs "
                                  "than the data (%i vs %i). Load anyways "
                                  "(epochs missing from the file will be "
                                  "accepted)?" % (ds.n_cases, self.n_epochs),
                                  "Load partial file?", wx.OK | wx.CANCEL |
                                  wx.CANCEL_DEFAULT | wx.ICON_WARNING)
            if cmd == wx.OK:
                n_missing = self.n_epochs - ds.n_cases
                tail = Dataset(info = ds.info)
                tail['trigger'] = Var(self.trigger[-n_missing:])
                tail['accept'] = Var([True], repeat=n_missing)
                tail['tag'] = Factor(['missing'], repeat=n_missing)
                ds = combine((ds, tail))
            else:
                raise RuntimeError("Unequal number of cases")

        if not np.all(ds[self.trigger.name] == self.trigger):
            app = get_app()
            cmd = app.message_box("The file contains different triggers from "
                                  "the data. Ignore?",
                                  "Ignore trigger mismatch?",
                                  wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT)
            if cmd == wx.OK:
                ds[self.trigger.name] = self.trigger
            else:
                raise RuntimeError("Trigger mismatch")

        accept = ds['accept']
        if 'rej_tag' in ds:
            tag = ds['rej_tag']
        else:
            tag = Factor([''], repeat=self.n_epochs, name='rej_tag')

        if INTERPOLATE_CHANNELS in ds:
            interpolate = ds[INTERPOLATE_CHANNELS]
            if not self.allow_interpolation and any(interpolate):
                app = get_app()
                cmd = app.message_box("The file contains channel interpolation "
                                      "instructions, but interpolation is "
                                      "disabled is the current session. Drop "
                                      "interpolation instructions?",
                                      "Clear Channel Interpolation "
                                      "Instructions?",
                                      wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT)
                if cmd == wx.OK:
                    for l in interpolate:
                        del l[:]
                else:
                    raise RuntimeError("File with interpolation when "
                                       "Interpolation is disabled")
        else:
            interpolate = Datalist([[]] * self.n_epochs, INTERPOLATE_CHANNELS,
                                   'strlist')

        if BAD_CHANNELS in ds.info:
            bad_channels = self.epochs.sensor.dimindex(ds.info[BAD_CHANNELS])
        else:
            bad_channels = []

        return accept, tag, interpolate, bad_channels

    def save(self):
        # find dest path
        _, ext = os.path.splitext(self.path)

        # create Dataset to save
        info = {BAD_CHANNELS: self.bad_channel_names}
        ds = Dataset((self.trigger, self.accept, self.tag, self.interpolate),
                     info=info)

        if ext == '.pickled':
            save.pickle(ds, self.path)
        elif ext == '.txt':
            ds.save_txt(self.path)
        else:
            raise ValueError("Unsupported extension: %r" % ext)

    def subscribe_to_bad_channels_change(self, callback):
        "callback()"
        self._bad_chs_change_subscriptions.append(callback)

    def subscribe_to_case_change(self, callback):
        "callback(index)"
        self._case_change_subscriptions.append(callback)


class Model(FileModel):
    """Manages a document as well as its history"""

    def clear(self):
        desc = "Clear"
        index = np.logical_not(self.doc.accept.x)
        old_tag = self.doc.tag[index]
        action = ChangeAction(desc, index, False, True, old_tag, 'clear')
        logger = getLogger(__name__)
        logger.info("Clearing %i rejections" % index.sum())
        self.history.do(action)

    def load(self, path):
        new_accept, new_tag, new_interpolate, new_bad_chs = self.doc.read_rej_file(path)

        # create load action
        action = ChangeAction("Load File", full_slice, self.doc.accept, new_accept,
                              self.doc.tag, new_tag, self.doc.path, path,
                              self.doc.bad_channels, new_bad_chs,
                              self.doc.interpolate, new_interpolate)
        self.history.do(action)
        self.history.register_save()

    def set_bad_channels(self, bad_channels, desc="Set bad channels"):
        "Set bad channels with a list of int"
        action = ChangeAction(desc, old_bad_chs=self.doc.bad_channels,
                              new_bad_chs=bad_channels)
        self.history.do(action)

    def set_case(self, i, state, tag=None, desc="Manual Change"):
        old_accept = self.doc.accept[i]
        if tag is None:
            old_tag = None
        else:
            old_tag = self.doc.tag[i]
        action = ChangeAction(desc, i, old_accept, state, old_tag, tag)
        self.history.do(action)

    def set_interpolation(self, case, ch_names):
        action = ChangeAction("Epoch %s interpolate %r" % (case, ', '.join(ch_names)),
                              case, old_interpolate=self.doc.interpolate[case],
                              new_interpolate=ch_names)
        self.history.do(action)

    def threshold(self, threshold=2e-12, method='abs'):
        """Find epochs based on a threshold criterion

        Parameters
        ----------
        threshold : scalar
            The threshold value. Examples: 1.25e-11 to detect saturated
            channels; 2e-12: for conservative MEG rejection.
        method : 'abs' | 'p2p'
            How to apply the threshold. With "abs", the threshold is applied to
            absolute values. With 'p2p' the threshold is applied to
            peak-to-peak values.

        Returns
        -------
        sub_threshold : array of bool
            True for all epochs in which the criterion is not reached (i.e.,
            epochs that should be accepted).
        """
        args = ', '.join(map(str, (threshold, method)))
        logger = getLogger(__name__)
        logger.info("Auto-reject trials: %s" % args)

        x = self.doc.good_data()
        if method == 'abs':
            x_max = x.abs().max(('time', 'sensor'))
            sub_threshold = x_max <= threshold
        elif method == 'p2p':
            p2p = x.max('time') - x.min('time')
            max_p2p = p2p.max('sensor')
            sub_threshold = max_p2p <= threshold
        else:
            raise ValueError("Invalid method: %r" % method)
        return sub_threshold

    def toggle_interpolation(self, case, ch_name):
        old_interpolate = self.doc.interpolate[case]
        new_interpolate = old_interpolate[:]
        if ch_name in new_interpolate:
            new_interpolate.remove(ch_name)
            desc = "Don't interpolate %s for %i" % (ch_name, case)
        else:
            new_interpolate.append(ch_name)
            desc = "Interpolate %s for %i" % (ch_name, case)
        action = ChangeAction(desc, case, old_interpolate=old_interpolate,
                              new_interpolate=new_interpolate)
        self.history.do(action)

    def update_bad_chs(self, bad_chs, interp, desc):
        if interp is None:
            index = old_interp = new_interp = None
        else:
            new_interp = self.doc.interpolate[:]
            new_interp._update_listlist(interp)
            index = np.flatnonzero(new_interp != self.doc.interpolate)
            if len(index):
                old_interp = self.doc.interpolate[index]
                new_interp = new_interp[index]
            else:
                index = old_interp = new_interp = None

        # find changed bad channels
        old_bad_chs = self.doc.bad_channel_names
        if bad_chs and any(ch not in old_bad_chs for ch in bad_chs):
            new_bad_chs = sorted(set(old_bad_chs).union(bad_chs))
        else:
            new_bad_chs = old_bad_chs = None

        action = ChangeAction(desc, index, old_bad_chs=old_bad_chs, new_bad_chs=new_bad_chs,
                              old_interpolate=old_interp, new_interpolate=new_interp)
        self.history.do(action)

    def update_rejection(self, new_accept, mark_good, mark_bad, desc, new_tag):
        # find changes
        if not mark_good:
            index = np.invert(new_accept)
        elif not mark_bad:
            index = new_accept
        else:
            index = None

        if index is None:
            index = new_accept != self.doc.accept.x
        else:
            np.logical_and(index, new_accept != self.doc.accept.x, index)
        index = np.flatnonzero(index)

        # construct action
        old_accept = self.doc.accept[index]
        new_accept = new_accept[index]
        old_tag = self.doc.tag[index]
        action = ChangeAction(desc, index, old_accept, new_accept, old_tag,
                              new_tag)
        self.history.do(action)


class Frame(FileFrame):
    """
    Epoch Rejection
    ===============

    Exclude bad epochs and interpolate or remove bad channels.

    * Use the `Bad Channels` button in the toolbar to exclude channels from
      analysis (use the `GA` button to plot the grand average and look for
      channels that are consistently bad).
    * Click the `Threshold` button to automatically reject epochs in which the
      signal exceeds a certain threshold.
    * Click on an epoch plot to toggle rejection of that epoch.
    * Press ``i`` on the keyboard to toggle channel interpolation for the
      channel that is closest to the cursor along the y-axis.
    * Press ``shift-i`` on the keyboard to edit a list of interpolated channels
      for the epoch under the cursor.


    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    right-arrow go to the next page
    left-arrow  go to the previous page
    b           butterfly plot of the epoch under the pointer
    c           pairwise sensor correlation plot or the current epoch
    t           topomap plot of the epoch/time point under the pointer
    i           interpolate the channel nearest to the pointer on the y-axis
    shift-i     open dialog to enter channels for interpolation
    =========== ============================================================
    """
    _doc_name = 'epoch selection'
    _name = "SelectEpochs"
    _title = "Select Epochs"

    def __init__(self, parent, model, nplots, topo, mean, vlim, color, lw, mark,
                 mcolor, mlw, antialiased, pos, size, allow_interpolation):
        """View object of the epoch selection GUI

        Parameters
        ----------
        parent : wx.Frame
            Parent window.
        others :
            See TerminalInterface constructor.
        """
        super(Frame, self).__init__(parent, pos, size, model)
        self.allow_interpolation = allow_interpolation

        # bind events
        self.doc.subscribe_to_case_change(self.CaseChanged)
        self.doc.subscribe_to_bad_channels_change(self.ShowPage)

        # setup figure canvas
        self.canvas = FigureCanvasPanel(self)
        self.figure = self.canvas.figure
        self.figure.set_facecolor('white')
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.025,
                                    top=.975, wspace=.1, hspace=.25)

        # Toolbar
        tb = self.InitToolbar()
        tb.AddSeparator()

        # --> select page
        txt = wx.StaticText(tb, -1, "Page:")
        tb.AddControl(txt)
        self.page_choice = wx.Choice(tb, -1)
        tb.AddControl(self.page_choice)
        tb.Bind(wx.EVT_CHOICE, self.OnPageChoice)

        # --> forward / backward
        self.back_button = tb.AddLabelTool(wx.ID_BACKWARD, "Back",
                                           Icon("tango/actions/go-previous"))
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        self.next_button = tb.AddLabelTool(wx.ID_FORWARD, "Next",
                                           Icon("tango/actions/go-next"))
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        tb.AddSeparator()

        # --> Bad Channels
        button = wx.Button(tb, ID.SET_BAD_CHANNELS, "Bad Channels")
        button.Bind(wx.EVT_BUTTON, self.OnSetBadChannels)
        tb.AddControl(button)

        # --> Thresholding
        button = wx.Button(tb, ID.THRESHOLD, "Threshold")
        button.Bind(wx.EVT_BUTTON, self.OnThreshold)
        tb.AddControl(button)

        # right-most part
        tb.AddStretchableSpace()

        # Grand-average plot
        button = wx.Button(tb, ID.GRAND_AVERAGE, "GA")
        button.SetHelpText("Plot the grand average of all accepted epochs")
        button.Bind(wx.EVT_BUTTON, self.OnPlotGrandAverage)
        tb.AddControl(button)

        # Info
        tb.AddLabelTool(wx.ID_INFO, 'Info', Icon("actions/info"))
        self.Bind(wx.EVT_TOOL, self.OnInfo, id=wx.ID_INFO)

        # --> Help
        self.InitToolbarTail(tb)
        tb.Realize()

        self.CreateStatusBar()

        # check plot parameters
        warnings = []
        if mark:
            mark, invalid = self.doc.epochs.sensor._normalize_sensor_names(mark, missing='return')
            if invalid:
                msg = ("Some channels specified to mark do not exist in the "
                       "data and will be ignored: " + ', '.join(invalid))
                wx.CallLater(1, wx.MessageBox, msg, "Invalid Channels in Mark",
                             wx.OK | wx.ICON_WARNING)

        # setup plot parameters
        self._vlims = find_fig_vlims([[self.doc.epochs]])
        if vlim is not None:
            for k in self._vlims:
                self._vlims[k] = (-vlim, vlim)
        self._mark = mark
        self._bfly_kwargs = {'color': color, 'lw': lw, 'mlw': mlw,
                             'antialiased': antialiased, 'vlims': self._vlims,
                             'mcolor': mcolor}
        self._topo_kwargs = {'vlims': self._vlims}
        self._SetLayout(nplots, topo, mean)

        # Bind Events ---
        self.canvas.mpl_connect('axes_leave_event', self.OnPointerLeaveAxes)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        self.canvas.mpl_connect('motion_notify_event', self.OnPointerMotion)

        # plot objects
        self._current_page_i = None
        self._epoch_idxs = None
        self._case_plots = None
        self._case_axes = None
        self._case_segs = None
        self._axes_by_idx = None
        self._topo_ax = None
        self._topo_plot_info_str = None
        self._topo_plot = None
        self._mean_plot = None

        # Finalize
        self.ShowPage(0)
        self.UpdateTitle()

    def CanBackward(self):
        return self._current_page_i > 0

    def CanForward(self):
        return self._current_page_i < self._n_pages - 1

    def CaseChanged(self, index):
        "updates the states of the segments on the current page"
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.doc.n_epochs
            index = xrange(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # update epoch plots
        axes = []
        for idx in index:
            if idx in self._axes_by_idx:
                ax = self._axes_by_idx[idx]
                ax_idx = ax.ax_idx
                h = self._case_plots[ax_idx]
                h.set_state(self.doc.accept[idx])
                # interpolated channels
                ch_index = h.epoch.sensor.channel_idx
                h.set_marked(INTERPOLATE_CHANNELS, [ch_index[ch] for ch in
                                                    self.doc.interpolate[idx]
                                                    if ch in ch_index])

                axes.append(ax)

        # update mean plot
        if self._plot_mean:
            self._mean_plot.set_data(self._get_page_mean_seg())
            axes.append(self._mean_ax)

        self.canvas.redraw(axes=axes)

    def GoToEpoch(self, i):
        for page, epochs in enumerate(self._segs_by_page):
            if i in epochs:
                break
        else:
            raise ValueError("Epoch not found: %r" % i)
        if page != self._current_page_i:
            self.SetPage(page)

    def MakeToolsMenu(self, menu):
        app = wx.GetApp()
        item = menu.Append(wx.ID_ANY, "Set Bad Channels",
                           "Specify bad channels for the whole file")
        app.Bind(wx.EVT_MENU, self.OnSetBadChannels, item)
        item = menu.Append(wx.ID_ANY, "Find Epochs by Threshold",
                           "Find epochs based in a specific threshold")
        app.Bind(wx.EVT_MENU, self.OnThreshold, item)
        item = menu.Append(wx.ID_ANY, "Find Bad Channels",
                           "Find bad channels using different criteria")
        app.Bind(wx.EVT_MENU, self.OnFindNoisyChannels, item)
        menu.AppendSeparator()
        item = menu.Append(wx.ID_ANY, "Plot Grand Average",
                           "Plot the grand average of all accepted epochs "
                           "(does not perform single-epoch channel "
                           "interpolation)")
        app.Bind(wx.EVT_MENU, self.OnPlotGrandAverage, item)
        item = menu.Append(wx.ID_ANY, "Info")
        app.Bind(wx.EVT_MENU, self.OnInfo, item)

    def OnBackward(self, event):
        "turns the page backward"
        self.SetPage(self._current_page_i - 1)

    def OnCanvasClick(self, event):
        "called by mouse clicks"
        ax = event.inaxes
        logger = getLogger(__name__)
        if ax:
            logger.debug("Canvas click at ax.ax_idx=%i", ax.ax_idx)
            if ax.ax_idx >= 0:
                idx = ax.epoch_idx
                state = not self.doc.accept[idx]
                tag = "manual"
                desc = "Epoch %i %s" % (idx, state)
                self.model.set_case(idx, state, tag, desc)
            elif ax.ax_idx == TOPO_PLOT:
                self.open_topomap()
        else:
            logger.debug("Canvas click outside axes")

    def OnCanvasKey(self, event):
        # GUI Control events
        if event.key == 'right':
            if self.CanForward():
                self.OnForward(None)
            return
        elif event.key == 'left':
            if self.CanBackward():
                self.OnBackward(None)
            return
        elif event.key == 'u':
            if self.CanUndo():
                self.OnUndo(None)
            return
        elif event.key == 'U':
            if self.CanRedo():
                self.OnRedo(None)
            return

        # plotting
        ax = event.inaxes
        if ax is None or ax.ax_idx == TOPO_PLOT:
            return
        elif ax.ax_idx > 0 and ax.epoch_idx == OUT_OF_RANGE:
            return
        elif event.key == 't':
            self.PlotTopomap(ax.ax_idx, event.xdata)
        elif event.key == 'b':
            self.PlotButterfly(ax.ax_idx)
        elif event.key == 'c':
            self.PlotCorrelation(ax.ax_idx)
        elif ax.ax_idx == MEAN_PLOT:
            return
        elif event.key == 'i':
            self.ToggleChannelInterpolation(ax, event)
        elif event.key == 'I':
            self.OnSetInterpolation(ax.epoch_idx)

    def OnFindNoisyChannels(self, event):
        dlg = FindNoisyChannelsDialog(self)
        if dlg.ShowModal() == wx.ID_OK:
            # Find bad channels
            flat, flat_average, mincorr = dlg.GetValues()
            if flat:
                flats = meeg.find_flat_epochs(self.doc.epochs, flat)
            else:
                flats = None

            if flat_average:
                flats_av = meeg.find_flat_evoked(self.doc.epochs, flat_average)
            else:
                flats_av = None

            if mincorr:
                noisies = meeg.find_noisy_channels(self.doc.epochs, mincorr)
            else:
                noisies = None

            # Apply
            if dlg.do_apply.GetValue():
                has_flats = flats and any(flats)
                has_noisies = noisies and any(noisies)
                if has_flats and has_noisies:
                    interp = flats[:]
                    interp._update_listlist(noisies)
                elif has_flats:
                    interp = flats
                elif has_noisies:
                    interp = noisies
                else:
                    interp = None
                self.model.update_bad_chs(flats_av, interp, "Find noisy channels")

            # Show Report
            if dlg.do_report.GetValue():
                doc = fmtxt.Section("Noisy Channels")
                doc.append("Total of %i epochs." % len(self.doc.epochs))
                if flat_average:
                    sec = doc.add_section("Flat in the average (<%s)" % flat_average)
                    if flats_av:
                        sec.add_paragraph(', '.join(flats_av))
                    else:
                        sec.add_paragraph("None.")
                if flat:
                    sec = doc.add_section("Flat Channels (<%s)" % flat)
                    sec.add_paragraph(format_epoch_list(flats))
                if mincorr:
                    sec = doc.add_section("Neighbor correlation < %s" % mincorr)
                    sec.add_paragraph(format_epoch_list(noisies))
                InfoFrame(self, "Noisy Channels", doc.get_html())

            dlg.StoreConfig()
        dlg.Destroy()

    def OnForward(self, event):
        "turns the page forward"
        self.SetPage(self._current_page_i + 1)

    def OnInfo(self, event):
        doc = fmtxt.Section("%i Epochs" % len(self.doc.epochs))

        # rejected epochs
        rejected = np.invert(self.doc.accept.x)
        sec = doc.add_section(_report.n_of(rejected.sum(), 'epoch') + ' rejected')
        if np.any(rejected):
            para = fmtxt.delim_list((fmtxt.Link(epoch, "epoch:%i" % epoch) for
                                     epoch in np.flatnonzero(rejected)))
            sec.add_paragraph(para)

        # bad channels
        heading = _report.n_of(len(self.doc.bad_channels), "bad channel", True)
        sec = doc.add_section(heading.capitalize())
        if self.doc.bad_channels:
            sec.add_paragraph(', '.join(self.doc.bad_channel_names))

        # interpolation by epochs
        if any(self.doc.interpolate):
            sec = doc.add_section("Interpolate channels")
            sec.add_paragraph(format_epoch_list(self.doc.interpolate, "Interpolation by epoch:"))
        else:
            doc.add_section("No channels interpolated in individual epochs")

        InfoFrame(self, "Rejection Info", doc.get_html())

    def OnPageChoice(self, event):
        "called by the page Choice control"
        page = event.GetSelection()
        self.SetPage(page)

    def OnPlotGrandAverage(self, event):
        self.PlotGrandAverage()

    def OnPointerLeaveAxes(self, event):
        sb = self.GetStatusBar()
        sb.SetStatusText("", 0)

    def OnPointerMotion(self, event):
        "update view on mouse pointer movement"
        ax = event.inaxes
        if not ax:
            return self.SetStatusText("")
        elif ax.ax_idx == TOPO_PLOT:
            return self.SetStatusText(self._topo_plot_info_str)
        elif ax.epoch_idx == OUT_OF_RANGE:
            return self.SetStatusText("")

        # compose status text
        x = ax.xaxis.get_major_formatter().format_data(event.xdata)
        y = ax.yaxis.get_major_formatter().format_data(event.ydata)
        desc = "Page average" if ax.ax_idx == MEAN_PLOT else "Epoch %i" % ax.epoch_idx
        status = "%s,  x = %s ms,  y = %s" % (desc, x, y)
        if ax.ax_idx >= 0:  # single trial plot
            interp = self.doc.interpolate[ax.epoch_idx]
            if interp:
                status += ",  interpolate %s" % ', '.join(interp)
        self.SetStatusText(status)

        # update topomap
        if self._plot_topo:
            tseg = self._get_ax_data(ax.ax_idx, event.xdata)
            if self._topo_plot:
                self._topo_plot.set_data([tseg])
            else:
                self._topo_plot = _ax_topomap(self._topo_ax, [tseg], False,
                                              **self._topo_kwargs)
            self.canvas.redraw(axes=[self._topo_ax])
            self._topo_plot_info_str = "Topomap: %s, t = %s ms" % (desc, x)

    def OnSetBadChannels(self, event):
        dlg = wx.TextEntryDialog(self, "Please enter bad channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set Bad "
                                 "Channels", ', '.join(self.doc.bad_channel_names))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names_in = filter(None, (s.strip() for s in
                                             dlg.GetValue().split(',')))
                    names = self.doc.epochs.sensor._normalize_sensor_names(names_in)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        bad_channels = self.doc.epochs.sensor.dimindex(names)
        self.model.set_bad_channels(bad_channels)

    def OnSetInterpolation(self, epoch):
        "Show Dialog for channel interpolation for this epoch (index)"
        old = self.doc.interpolate[epoch]
        dlg = wx.TextEntryDialog(self, "Please enter channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set "
                                 "Channels for Interpolation", ', '.join(old))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names = filter(None, (s.strip() for s in dlg.GetValue().split(',')))
                    new = self.doc.epochs.sensor._normalize_sensor_names(names)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        if new != old:
            self.model.set_interpolation(epoch, new)

    def OnSetLayout(self, event):
        caption = "Set Plot Layout"
        msg = ("Number of epoch plots for square layout (e.g., '10') or \n"
               "exact n_rows and n_columns (e.g., '5 4'). Add 'nomean' to \n"
               "turn off plotting the page mean at the bottom right (e.g., "
               "'3 nomean').")
        default = ""
        dlg = wx.TextEntryDialog(self, msg, caption, default)
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                nplots = None
                topo = True
                mean = True
                err = []

                # separate options from layout
                value = dlg.GetValue()
                items = value.split(' ')
                options = []
                while not items[-1].isdigit():
                    options.append(items.pop(-1))

                # extract options
                for option in options:
                    if option == 'nomean':
                        mean = False
                    elif option == 'notopo':
                        topo = False
                    else:
                        err.append('Unknown option: "%s"' % option)

                # extract layout info
                if len(items) == 1 and items[0].isdigit():
                    nplots = int(items[0])
                elif len(items) == 2 and all(item.isdigit() for item in items):
                    nplots = tuple(int(item) for item in items)
                else:
                    value_ = ' '.join(items)
                    err = 'Invalid layout specification: "%s"' % value_

                # if all ok: break
                if nplots and not err:
                    break

                # error
                caption = 'Invalid Layout Entry: "%s"' % value
                err.append('Please read the instructions and try again.')
                msg = '\n'.join(err)
                style = wx.OK | wx.ICON_ERROR
                wx.MessageBox(msg, caption, style)
            else:
                dlg.Destroy()
                return

        dlg.Destroy()
        self.SetLayout(nplots, topo, mean)

    def OnSetMarkedChannels(self, event):
        "mark is represented in sensor names"
        dlg = wx.TextEntryDialog(self, "Please enter channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set Marked"
                                 "Channels", ', '.join(self._mark))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names_in = filter(None, (s.strip() for s in
                                             dlg.GetValue().split(',')))
                    names = self.doc.epochs.sensor._normalize_sensor_names(names_in)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        self.SetPlotStyle(mark=names)

    def OnSetVLim(self, event):
        default = str(self._vlims.values()[0][1])
        dlg = wx.TextEntryDialog(self, "New Y-axis limit:", "Set Y-Axis Limit",
                                 default)

        if dlg.ShowModal() == wx.ID_OK:
            value = dlg.GetValue()
            try:
                vlim = abs(float(value))
            except Exception as exception:
                msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                       wx.OK | wx.ICON_ERROR)
                msg.ShowModal()
                msg.Destroy()
                raise
            self.SetVLim(vlim)
        dlg.Destroy()

    def OnThreshold(self, event):
        dlg = ThresholdDialog(self)
        if dlg.ShowModal() == wx.ID_OK:
            threshold = dlg.GetThreshold()
            method = dlg.GetMethod()

            sub_threshold = self.model.threshold(threshold, method)
            if dlg.do_apply.GetValue():
                self.model.update_rejection(sub_threshold, dlg.GetMarkBelow(),
                                            dlg.GetMarkAbove(),
                                            "Threshold-%s" % method,
                                            "%s_%s" % (method, threshold))
            if dlg.do_report.GetValue():
                rejected = np.invert(sub_threshold)

                doc = fmtxt.Section("Threshold")
                doc.append("%s at %s: %i epochs" % (method, threshold, rejected.sum()))
                if np.any(rejected):
                    para = fmtxt.delim_list((fmtxt.Link(epoch, "epoch:%i" % epoch) for
                                             epoch in np.flatnonzero(rejected)))
                    doc.add_paragraph(para)
                InfoFrame(self, "Rejection Info", doc.get_html())
            dlg.StoreConfig()
        dlg.Destroy()

    def OnUpdateUIBackward(self, event):
        event.Enable(self.CanBackward())

    def OnUpdateUIForward(self, event):
        event.Enable(self.CanForward())

    def OnUpdateUISetLayout(self, event):
        event.Enable(True)

    def OnUpdateUISetMarkedChannels(self, event):
        event.Enable(True)

    def OnUpdateUISetVLim(self, event):
        event.Enable(True)

    def PlotCorrelation(self, ax_index):
        if ax_index == MEAN_PLOT:
            seg = self._mean_seg
            name = 'Page Mean Neighbor Correlation'
        else:
            epoch_idx = self._epoch_idxs[ax_index]
            seg = self._case_segs[ax_index]
            name = 'Epoch %i Neighbor Correlation' % epoch_idx
        plot.Topomap(corr(seg, name=name), sensorlabels='name')

    def PlotButterfly(self, ax_index):
        epoch = self._get_ax_data(ax_index)
        plot.TopoButterfly(epoch, vmax=self._vlims)

    def PlotGrandAverage(self):
        epoch = self.doc.get_grand_average()
        plot.TopoButterfly(epoch)

    def PlotTopomap(self, ax_index, time):
        tseg = self._get_ax_data(ax_index, time)
        plot.Topomap(tseg, vmax=self._vlims, sensorlabels='name', w=8,
                     title=tseg.name)

    def SetLayout(self, nplots=(6, 6), topo=True, mean=True):
        """Determine the layout of the Epochs canvas

        Parameters
        ----------
        nplots : int | tuple of 2 int
            Number of epoch plots per page. Can be an ``int`` to produce a
            square layout with that many epochs, or an ``(n_rows, n_columns)``
            tuple.
        topo : bool
            Show a topomap plot of the time point under the mouse cursor.
        mean : bool
            Show a plot of the page mean at the bottom right of the page.
        """
        self._SetLayout(nplots, topo, mean)
        self.ShowPage(0)

    def _SetLayout(self, nplots, topo, mean):
        if topo is None:
            topo = self.config.ReadBool('Layout/show_topo', True)
        else:
            topo = bool(topo)
            self.config.WriteBool('Layout/show_topo', topo)

        if mean is None:
            mean = self.config.ReadBool('Layout/show_mean', True)
        else:
            mean = bool(mean)
            self.config.WriteBool('Layout/show_mean', mean)

        if nplots is None:
            nrow = self.config.ReadInt('Layout/n_rows', 6)
            ncol = self.config.ReadInt('Layout/n_cols', 6)
            nax = ncol * nrow
            n_per_page = nax - bool(topo) - bool(mean)
        else:
            if isinstance(nplots, int):
                if nplots == 1:
                    mean = False
                elif nplots < 1:
                    raise ValueError("nplots needs to be >= 1; got %r" % nplots)
                nax = nplots + bool(mean) + bool(topo)
                nrow = math.ceil(math.sqrt(nax))
                ncol = int(math.ceil(nax / nrow))
                nrow = int(nrow)
                n_per_page = nplots
            else:
                nrow, ncol = nplots
                nax = ncol * nrow
                if nax == 1:
                    mean = False
                    topo = False
                elif nax == 2:
                    mean = False
                elif nax < 1:
                    err = ("nplots=%s: Need at least one plot." % str(nplots))
                    raise ValueError(err)
                n_per_page = nax - bool(topo) - bool(mean)
            self.config.WriteInt('Layout/n_rows', nrow)
            self.config.WriteInt('Layout/n_cols', ncol)
        self.config.Flush()

        self._plot_mean = mean
        self._plot_topo = topo

        # prepare segments
        n = self.doc.n_epochs
        self._nplots = (nrow, ncol)
        self._n_per_page = n_per_page
        self._n_pages = n_pages = int(math.ceil(n / n_per_page))

        # get a list of IDS for each page
        self._segs_by_page = []
        for i in xrange(n_pages):
            start = i * n_per_page
            stop = min((i + 1) * n_per_page, n)
            self._segs_by_page.append(range(start, stop))

        # update page selector
        pages = []
        for i in xrange(n_pages):
            istart = self._segs_by_page[i][0]
            if i == n_pages - 1:
                pages.append('%i: %i..%i' % (i, istart, self.doc.n_epochs))
            else:
                pages.append('%i: %i...' % (i, istart))
        self.page_choice.SetItems(pages)

    def SetPlotStyle(self, **kwargs):
        """Select channels to mark in the butterfly plots.

        Parameters
        ----------
        color : None | matplotlib color
            Color for primary data (default is black).
        lw : scalar
            Linewidth for normal sensor plots.
        mark : None | str | list of str
            Sensors to plot as individual traces with a separate color.
        mcolor : matplotlib color
            Color for marked traces.
        mlw : scalar
            Line width for marked sensor plots.
        antialiased : bool
            Perform Antialiasing on epoch plots (associated with a minor speed
            cost).
        """
        self._SetPlotStyle(**kwargs)
        self.ShowPage()

    def _SetPlotStyle(self, **kwargs):
        "See .SetPlotStyle()"
        for key, value in kwargs.iteritems():
            if key == 'vlims':
                err = ("%r is an invalid keyword argument for this function"
                       % key)
                raise TypeError(err)
            elif key == 'mark':
                self._mark = value
            elif key in self._bf_kwargs:
                self._bfly_kwargs[key] = value
            else:
                raise KeyError(repr(key))

    def SetVLim(self, vlim):
        """Set the value limits (butterfly plot y axes and topomap colormaps)

        Parameters
        ----------
        vlim : scalar | (scalar, scalar)
            For symmetric limits the positive vmax, for asymmetric limits a
            (vmin, vmax) tuple.
        """
        for p in self._case_plots:
            p.set_ylim(vlim)
        if self._mean_plot:
            self._mean_plot.set_ylim(vlim)
        if self._topo_plot:
            self._topo_plot.set_vlim(vlim)

        if np.isscalar(vlim):
            vlim = (-vlim, vlim)
        for key in self._vlims:
            self._vlims[key] = vlim
        self.canvas.draw()

    def _page_change(self, page):
        "Operations common to page change"
        self._current_page_i = page
        self.page_choice.Select(page)
        self._epoch_idxs = self._segs_by_page[page]

    def SetPage(self, page):
        "Change the page that is displayed without redrawing"
        self._page_change(page)

        self._case_segs = []
        self._axes_by_idx = {}

        for i, epoch_idx in enumerate(self._epoch_idxs):
            ax = self._case_axes[i]
            h = self._case_plots[i]
            case = self.doc.get_epoch(epoch_idx, 'Epoch %i' % epoch_idx)
            h.set_data(case, epoch_idx)
            h.set_state(self.doc.accept[epoch_idx])
            chs = [case.sensor.channel_idx[ch]
                   for ch in self.doc.interpolate[epoch_idx]
                   if ch in case.sensor.channel_idx]
            h.set_marked(INTERPOLATE_CHANNELS, chs)
            h.set_visible()
            ax.epoch_idx = epoch_idx

            # store objects
            ax.epoch_idx = epoch_idx
            self._case_segs.append(case)
            self._axes_by_idx[epoch_idx] = ax

        # hide lines axes without data
        if len(self._epoch_idxs) < len(self._case_axes):
            for i in xrange(len(self._epoch_idxs), len(self._case_plots)):
                self._case_plots[i].set_visible(False)
                self._case_axes[i].epoch_idx = OUT_OF_RANGE

        # update mean plot
        if self._plot_mean:
            self._mean_plot.set_data(self._get_page_mean_seg())

        self.canvas.draw()

    def ShowPage(self, page=None):
        "Dislay a specific page (start counting with 0)"
        wx.BeginBusyCursor()
        logger = getLogger(__name__)
        t0 = time.time()
        if page is None:
            page = self._current_page_i
        else:
            self._page_change(page)

        self.figure.clf()
        nrow, ncol = self._nplots

        # formatters
        t_formatter, t_label = find_axis_params_dim(self.doc.epochs.time, True)
        y_formatter, y_label = find_axis_params_data(self.doc.epochs, True)

        # segment plots
        self._case_plots = []
        self._case_axes = []
        self._case_segs = []
        self._axes_by_idx = {}
        mark = None
        for i, epoch_idx in enumerate(self._epoch_idxs):
            case = self.doc.get_epoch(epoch_idx, 'Epoch %i' % epoch_idx)
            if mark is None:
                mark = [case.sensor.channel_idx[ch] for ch in self._mark
                        if ch in case.sensor.channel_idx]
            state = self.doc.accept[epoch_idx]
            ax = self.figure.add_subplot(nrow, ncol, i + 1, xticks=[0], yticks=[])
            h = _ax_bfly_epoch(ax, case, mark, state, epoch_idx, **self._bfly_kwargs)
            # mark interpolated channels
            if self.doc.interpolate[epoch_idx]:
                chs = [case.sensor.channel_idx[ch]
                       for ch in self.doc.interpolate[epoch_idx]
                       if ch in case.sensor.channel_idx]
                h.set_marked(INTERPOLATE_CHANNELS, chs)
            # mark eye tracker artifacts
            if self.doc.blink is not None:
                _plt_bin_nuts(ax, self.doc.blink[epoch_idx],
                              color=(0.99, 0.76, 0.21))
            # formatters
            ax.xaxis.set_major_formatter(t_formatter)
            ax.yaxis.set_major_formatter(y_formatter)

            # store objects
            ax.ax_idx = i
            ax.epoch_idx = epoch_idx
            self._case_plots.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(case)
            self._axes_by_idx[epoch_idx] = ax

        # mean plot
        if self._plot_mean:
            plot_i = nrow * ncol
            self._mean_ax = ax = self.figure.add_subplot(nrow, ncol, plot_i)
            ax.ax_idx = MEAN_PLOT
            self._mean_seg = self._get_page_mean_seg()
            self._mean_plot = _ax_bfly_epoch(ax, self._mean_seg, mark, 'Page Mean',
                                             **self._bfly_kwargs)

            # formatters
            ax.xaxis.set_major_formatter(t_formatter)
            ax.yaxis.set_major_formatter(y_formatter)

        # topomap
        if self._plot_topo:
            plot_i = nrow * ncol - self._plot_mean
            self._topo_ax = self.figure.add_subplot(nrow, ncol, plot_i)
            self._topo_ax.ax_idx = TOPO_PLOT
            self._topo_ax.set_axis_off()
            self._topo_plot = None
            self._topo_plot_info_str = ""

        self.canvas.draw()
        self.canvas.store_canvas()

        dt = time.time() - t0
        logger.debug('Page draw took %.1f seconds.', dt)
        wx.EndBusyCursor()

    def ToggleChannelInterpolation(self, ax, event):
        if not self.allow_interpolation:
            wx.MessageBox("Interpolation is disabled for this session",
                          "Interpolation disabled", wx.OK)
            return
        plt = self._case_plots[ax.ax_idx]
        locs = plt.epoch.sub(time=event.xdata).x
        sensor = np.argmin(np.abs(locs - event.ydata))
        sensor_name = plt.epoch.sensor.names[sensor]
        self.model.toggle_interpolation(ax.epoch_idx, sensor_name)

    def _get_page_mean_seg(self):
        page_segments = self._segs_by_page[self._current_page_i]
        page_index = np.zeros(self.doc.n_epochs, dtype=bool)
        page_index[page_segments] = True
        index = np.logical_and(page_index, self.doc.accept.x)
        mseg = self.doc.get_epoch(index, "Page Average").mean('case')
        return mseg

    def _get_ax_data(self, ax_index, time=None):
        if ax_index == MEAN_PLOT:
            seg = self._mean_seg
            epoch_name = 'Page Average'
            sensor_idx = None
        elif ax_index >= 0:
            epoch_idx = self._epoch_idxs[ax_index]
            epoch_name = 'Epoch %i' % epoch_idx
            seg = self._case_segs[ax_index]
            sensor_idx = self.doc.good_sensor_index(epoch_idx)
        else:
            raise ValueError("Invalid ax_index: %s" % ax_index)

        if time is not None:
            name = '%s, %i ms' % (epoch_name, 1e3 * time)
            return seg.sub(time=time, sensor=sensor_idx, name=name)
        elif sensor_idx is None:
            return seg
        else:
            return seg.sub(sensor=sensor_idx)


class TerminalInterface(object):
    def __init__(self, ds, data='meg', accept='accept', blink='blink',
                 tag='rej_tag', trigger='trigger',
                 path=None, nplots=None, topo=None, mean=None,
                 vlim=None, color='k', lw=0.5, mark=[], mcolor='r', mlw=0.8,
                 antialiased=True, pos=None, size=None,
                 allow_interpolation=True):
        # Documented in eelbrain.gui
        bad_chs = None
        self.doc = Document(ds, data, accept, blink, tag, trigger, path,
                            bad_chs, allow_interpolation)
        self.model = Model(self.doc)
        self.history = self.model.history

        app = get_app()

        self.frame = Frame(None, self.model, nplots, topo, mean, vlim, color,
                           lw, mark, mcolor, mlw, antialiased, pos, size,
                           allow_interpolation)
        self.frame.Show()
        app.SetTopWindow(self.frame)
        if not app.IsMainLoopRunning():
            app.MainLoop()


class FindNoisyChannelsDialog(EelbrainDialog):
    def __init__(self, parent, *args, **kwargs):
        EelbrainDialog.__init__(self, parent, wx.ID_ANY, "Find Noisy Channels",
                                *args, **kwargs)
        # load config
        config = parent.config
        flat = config.ReadFloat("FindNoisyChannels/flat", 1e-13)
        flat_average = config.ReadFloat("FindNoisyChannels/flat_average", 1e-14)
        corr = config.ReadFloat("FindNoisyChannels/mincorr", 0.35)
        do_apply = config.ReadBool("FindNoisyChannels/do_apply", True)
        do_report = config.ReadBool("FindNoisyChannels/do_report", True)

        # construct layout
        sizer = wx.BoxSizer(wx.VERTICAL)

        # flat channels
        sizer.Add(wx.StaticText(self, label="Threshold for flat channels:"))
        msg = ("Invalid entry for flat channel threshold: {value}. Please "
               "specify a number > 0.")
        validator = REValidator(POS_FLOAT_PATTERN, msg, False)
        ctrl = wx.TextCtrl(self, value=str(flat), validator=validator)
        ctrl.SetHelpText("A channel that does not deviate from 0 by more than "
                         "this value is considered flat.")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.flat_ctrl = ctrl

        # flat channels in average
        sizer.Add(wx.StaticText(self, label="Threshold for flat channels in average:"))
        msg = ("Invalid entry for average flat channel threshold: {value}. Please "
               "specify a number > 0.")
        validator = REValidator(POS_FLOAT_PATTERN, msg, False)
        ctrl = wx.TextCtrl(self, value=str(flat_average), validator=validator)
        ctrl.SetHelpText("A channel that does not deviate from 0 by more than "
                         "this value in the average of all epochs is "
                         "considered flat.")
        sizer.Add(ctrl)
        self.flat_average_ctrl = ctrl

        # Correlation
        sizer.Add(wx.StaticText(self, label="Threshold for channel to neighbor correlation:"))
        msg = ("Invalid entry for channel neighbor correlation: {value}. Please "
               "specify a number > 0.")
        validator = REValidator(POS_FLOAT_PATTERN, msg, False)
        ctrl = wx.TextCtrl(self, value=str(corr), validator=validator)
        ctrl.SetHelpText("A channel is considered noisy if the average of the "
                         "correlation with its neighbors is smaller than this "
                         "value.")
        sizer.Add(ctrl)
        self.mincorr_ctrl = ctrl

        # output
        sizer.AddSpacer(4)
        sizer.Add(wx.StaticText(self, label="Output"))
        self.do_report = wx.CheckBox(self, wx.ID_ANY, "Show Report")
        self.do_report.SetValue(do_report)
        sizer.Add(self.do_report)
        self.do_apply = wx.CheckBox(self, wx.ID_ANY, "Apply")
        self.do_apply.SetValue(do_apply)
        sizer.Add(self.do_apply)

        # default button
        sizer.AddSpacer(4)
        btn = wx.Button(self, wx.ID_DEFAULT, "Default Settings")
        sizer.Add(btn, border=2)
        btn.Bind(wx.EVT_BUTTON, self.OnSetDefault)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btn.Bind(wx.EVT_BUTTON, self.OnOK)
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def GetValues(self):
        return (float(self.flat_ctrl.GetValue()),
                float(self.flat_average_ctrl.GetValue()),
                float(self.mincorr_ctrl.GetValue()))

    def OnOK(self, event):
        if self.do_report.GetValue() or self.do_apply.GetValue():
            event.Skip()
        else:
            wx.MessageBox("Specify at least one action (report or apply)",
                          "No Command Selected", wx.ICON_EXCLAMATION)

    def OnSetDefault(self, event):
        self.flat_ctrl.SetValue('1e-13')
        self.flat_average_ctrl.SetValue('1e-14')
        self.mincorr_ctrl.SetValue('0.35')

    def StoreConfig(self):
        config = self.Parent.config
        config.WriteFloat("FindNoisyChannels/flat",
                          float(self.flat_ctrl.GetValue()))
        config.WriteFloat("FindNoisyChannels/flat_average",
                          float(self.flat_average_ctrl.GetValue()))
        config.WriteFloat("FindNoisyChannels/mincorr",
                          float(self.mincorr_ctrl.GetValue()))
        config.WriteBool("FindNoisyChannels/do_report",
                         self.do_report.GetValue())
        config.WriteBool("FindNoisyChannels/do_apply",
                         self.do_apply.GetValue())
        config.Flush()


class ThresholdDialog(EelbrainDialog):

    _methods = (('absolute', 'abs'),
                ('peak-to-peak', 'p2p'))

    def __init__(self, parent):
        title = "Threshold Criterion Rejection"
        wx.Dialog.__init__(self, parent, wx.ID_ANY, title)

        # load config
        config = parent.config
        method = config.Read("Threshold/method", "p2p")
        mark_above = config.ReadBool("Threshold/mark_above", True)
        mark_below = config.ReadBool("Threshold/mark_below", False)
        threshold = config.ReadFloat("Threshold/threshold", 2e-12)
        do_apply = config.ReadBool("Threshold/do_apply", True)
        do_report = config.ReadBool("Threshold/do_report", True)

        # construct layout
        sizer = wx.BoxSizer(wx.VERTICAL)

        txt = "Mark epochs based on a threshold criterion"
        ctrl = wx.StaticText(self, wx.ID_ANY, txt)
        sizer.Add(ctrl)

        choices = tuple(m[0] for m in self._methods)
        ctrl = wx.RadioBox(self, wx.ID_ANY, "Method", choices=choices)
        selection = [m[1] for m in self._methods].index(method)
        ctrl.SetSelection(selection)
        sizer.Add(ctrl)
        self.method_ctrl = ctrl

        ctrl = wx.CheckBox(self, wx.ID_ANY, "Reject all epochs exceeding the "
                           "threshold")
        ctrl.SetValue(mark_above)
        sizer.Add(ctrl)
        self.mark_above = ctrl

        ctrl = wx.CheckBox(self, wx.ID_ANY, "Accept all epochs not exceeding "
                           "the threshold")
        ctrl.SetValue(mark_below)
        sizer.Add(ctrl)
        self.mark_below = ctrl

        msg = ("Invalid entry for threshold: {value}. Need a floating\n"
               "point number.")
        validator = REValidator(FLOAT_PATTERN, msg, False)
        ctrl = wx.TextCtrl(self, wx.ID_ANY, str(threshold),
                           validator=validator)
        ctrl.SetHelpText("Threshold value (positive scalar)")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.threshold_ctrl = ctrl

        # output
        sizer.AddSpacer(4)
        sizer.Add(wx.StaticText(self, label="Output"))
        self.do_report = wx.CheckBox(self, wx.ID_ANY, "Show Report")
        self.do_report.SetValue(do_report)
        sizer.Add(self.do_report)
        self.do_apply = wx.CheckBox(self, wx.ID_ANY, "Apply")
        self.do_apply.SetValue(do_apply)
        sizer.Add(self.do_apply)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btn.Bind(wx.EVT_BUTTON, self.OnOK)
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def GetMarkAbove(self):
        return self.mark_above.IsChecked()

    def GetMarkBelow(self):
        return self.mark_below.IsChecked()

    def GetMethod(self):
        index = self.method_ctrl.GetSelection()
        value = self._methods[index][1]
        return value

    def GetThreshold(self):
        text = self.threshold_ctrl.GetValue()
        value = float(text)
        return value

    def OnOK(self, event):
        if not (self.mark_above.GetValue() or self.mark_below.GetValue()):
            wx.MessageBox("Specify at least one action (reject or accept epochs)",
                          "No Command Selected", wx.ICON_EXCLAMATION)
        if not (self.do_report.GetValue() or self.do_apply.GetValue()):
            wx.MessageBox("Specify at least one action (report or apply)",
                          "No Command Selected", wx.ICON_EXCLAMATION)
        else:
            event.Skip()

    def StoreConfig(self):
        config = self.Parent.config
        config.Write("Threshold/method", self.GetMethod())
        config.WriteBool("Threshold/mark_above", self.GetMarkAbove())
        config.WriteBool("Threshold/mark_below", self.GetMarkBelow())
        config.WriteFloat("Threshold/threshold", self.GetThreshold())
        config.WriteBool("Threshold/do_report", self.do_report.GetValue())
        config.WriteBool("Threshold/do_apply", self.do_apply.GetValue())
        config.Flush()


class InfoFrame(HTMLFrame):

    def OpenURL(self, url):
        m = re.match('^(\w+):(\w+)$', url)
        if m:
            kind, address = m.groups()
            if kind == 'epoch':
                self.Parent.GoToEpoch(int(address))
            else:
                raise NotImplementedError("%s URL" % kind)
        else:
            raise ValueError("Invalid link URL: %r" % url)
