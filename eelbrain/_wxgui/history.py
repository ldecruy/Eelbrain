'''History for wx GUIs'''
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from logging import getLogger
import os

import wx

from .._wxutils import ID
from .help import show_help_txt
from .frame import EelbrainFrame


class Action(object):

    def do(self, doc):
        raise NotImplementedError

    def undo(self, doc):
        raise NotImplementedError


class History():
    """The history as a list of action objects

    Public interface
    ----------------
    can_redo() : bool
        Whether the history can redo an action.
    can_undo() : bool
        Whether the history can redo an action.
    do(action)
        perform a action
    is_saved() : bool
        Whether the current state is saved
    redo()
        Redo the latest undone action.
    ...
    """

    def __init__(self, doc):
        self.doc = doc
        self._history = []
        self._saved_change_subscriptions = []
        # point to last executed action (always < 0)
        self._last_action_idx = -1
        # point to action after which we saved ( > 0 if ever saved)
        self._saved_idx = -1

    def can_redo(self):
        return self._last_action_idx < -1

    def can_undo(self):
        return len(self._history) + self._last_action_idx >= 0

    def do(self, action):
        logger = getLogger(__name__)
        logger.debug("Do action: %s", action.desc)
        was_saved = self.is_saved()
        action.do(self.doc)
        if self._last_action_idx < -1:
            # discard alternate future
            self._history = self._history[:self._last_action_idx + 1]
            self._last_action_idx = -1
            if self._saved_idx >= len(self._history):
                self._saved_idx = -1
        self._history.append(action)
        self._process_saved_change(was_saved)

    def _process_saved_change(self, was_saved):
        """Process a state change in whether all changes are saved

        Parameters
        ----------
        was_saved : bool
            Whether all changes were saved before the current change happened.
        """
        is_saved = self.is_saved()
        if is_saved != was_saved:
            self.doc.saved = is_saved
            for func in self._saved_change_subscriptions:
                func()

    def is_saved(self):
        """Determine whether the document is saved

        Returns
        -------
        is_saved : bool
            Whether the document is saved (i.e., contains no unsaved changes).
        """
        current_index = len(self._history) + self._last_action_idx
        if current_index == -1 and self._saved_idx < 0:
            return True  # no actions and never saved
        return self._saved_idx == current_index

    def redo(self):
        was_saved = self.is_saved()
        if self._last_action_idx == -1:
            raise RuntimeError("We are at the tip of the history")
        action = self._history[self._last_action_idx + 1]
        logger = getLogger(__name__)
        logger.debug("Redo action: %s", action.desc)
        action.do(self.doc)
        self._last_action_idx += 1
        self._process_saved_change(was_saved)

    def register_save(self):
        "Notify the history that the document is saved at the current state"
        was_saved = self.is_saved()
        self._saved_idx = len(self._history) + self._last_action_idx
        self._process_saved_change(was_saved)

    def subscribe_to_saved_change(self, callback):
        "callback(saved)"
        self._saved_change_subscriptions.append(callback)

    def undo(self):
        was_saved = self.is_saved()
        if -self._last_action_idx > len(self._history):
            raise RuntimeError("We are at the beginning of the history")
        action = self._history[self._last_action_idx]
        logger = getLogger(__name__)
        logger.debug("Undo action: %s", action.desc)
        action.undo(self.doc)
        self._last_action_idx -= 1
        self._process_saved_change(was_saved)


class FileDocument(object):
    """Represent a file"""
    def __init__(self, path):
        self.saved = True  # managed by the history
        self.path = path
        self._path_change_subscriptions = []

    def set_path(self, path):
        self.path = path
        for func in self._path_change_subscriptions:
            func()

    def subscribe_to_path_change(self, callback):
        "callback(path)"
        self._path_change_subscriptions.append(callback)


class FileModel(object):
    """Manages a document as well as its history"""

    def __init__(self, doc):
        self.doc = doc
        self.history = History(doc)

    def load(self, path):
        raise NotImplementedError

    def save(self):
        self.doc.save()
        self.history.register_save()

    def save_as(self, path):
        self.doc.set_path(path)
        self.save()


class FileFrame(EelbrainFrame):
    _doc_name = 'document'
    _name = 'Default'
    _title = 'Title'
    _wildcard = ("Tab Separated Text (*.txt)|*.txt|"
                 "Pickle (*.pickled)|*.pickled")

    def __init__(self, parent, pos, size, model):
        """View object of the epoch selection GUI

        Parameters
        ----------
        parent : wx.Frame
            Parent window.
        others :
            See TerminalInterface constructor.
        """
        config = wx.Config("Eelbrain")
        config.SetPath(self._name)

        if pos is None:
            pos = (config.ReadInt("pos_horizontal", -1),
                   config.ReadInt("pos_vertical", -1))
        else:
            pos_h, pos_v = pos
            config.WriteInt("pos_horizontal", pos_h)
            config.WriteInt("pos_vertical", pos_v)
            config.Flush()

        if size is None:
            size = (config.ReadInt("size_width", 800),
                    config.ReadInt("size_height", 600))
        else:
            w, h = pos
            config.WriteInt("size_width", w)
            config.WriteInt("size_height", h)
            config.Flush()

        super(FileFrame, self).__init__(parent, -1, self._title, pos, size)
        self.config = config
        self.model = model
        self.doc = model.doc
        self.history = model.history

        # Bind Events ---
        self.doc.subscribe_to_path_change(self.UpdateTitle)
        self.history.subscribe_to_saved_change(self.UpdateTitle)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_TOOL, self.OnOpen, id=ID.OPEN)
        self.Bind(wx.EVT_TOOL, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_TOOL, self.OnSaveAs, id=wx.ID_SAVEAS)

    def CanRedo(self):
        return self.history.can_redo()

    def CanSave(self):
        return bool(self.doc.path)

    def CanUndo(self):
        return self.history.can_undo()

    def OnClear(self, event):
        self.model.clear()

    def OnClose(self, event):
        "Ask to save unsaved changes"
        if event.CanVeto() and not self.history.is_saved():
            msg = ("The current document has unsaved changes. Would you like "
                   "to save them?")
            cap = "Saved Unsaved Changes?"
            style = wx.YES | wx.NO | wx.CANCEL | wx.YES_DEFAULT
            cmd = wx.MessageBox(msg, cap, style)
            if cmd == wx.YES:
                if self.OnSave(event) != wx.ID_OK:
                    event.Veto()
                    return
            elif cmd == wx.CANCEL:
                event.Veto()
                return
            elif cmd != wx.NO:
                raise RuntimeError("Unknown answer: %r" % cmd)

        logger = getLogger(__name__)
        logger.debug("%s.OnClose(), saving window properties...",
                     self.__class__.__name__)
        pos_h, pos_v = self.GetPosition()
        w, h = self.GetSize()

        self.config.WriteInt("pos_horizontal", pos_h)
        self.config.WriteInt("pos_vertical", pos_v)
        self.config.WriteInt("size_width", w)
        self.config.WriteInt("size_height", h)
        self.config.Flush()

        event.Skip()

    def OnHelp(self, event):
        show_help_txt(self.__doc__, self, self._title)

    def OnOpen(self, event):
        msg = ("Load the %s from a file." % self._doc_name)
        if self.doc.path:
            default_dir, default_name = os.path.split(self.doc.path)
        else:
            default_dir = ''
            default_name = ''
        dlg = wx.FileDialog(self, msg, default_dir, default_name,
                            self._wildcard, wx.FD_OPEN)
        rcode = dlg.ShowModal()
        dlg.Destroy()

        if rcode != wx.ID_OK:
            return rcode

        path = dlg.GetPath()
        try:
            self.model.load(path)
        except Exception as ex:
            msg = str(ex)
            title = "Error Loading %s" % self._doc_name.capitalize()
            wx.MessageBox(msg, title, wx.ICON_ERROR)
            raise

    def OnRedo(self, event):
        self.history.redo()

    def OnSave(self, event):
        if self.doc.path:
            self.model.save()
            return wx.ID_OK
        else:
            return self.OnSaveAs(event)

    def OnSaveAs(self, event):
        msg = ("Save the %s to a file." % self._doc_name)
        if self.doc.path:
            default_dir, default_name = os.path.split(self.doc.path)
        else:
            default_dir = ''
            default_name = ''

        dlg = wx.FileDialog(self, msg, default_dir, default_name,
                            self._wildcard, wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        rcode = dlg.ShowModal()
        if rcode == wx.ID_OK:
            path = dlg.GetPath()
            self.model.save_as(path)

        dlg.Destroy()
        return rcode

    def OnUndo(self, event):
        self.history.undo()

    def OnUpdateUIOpen(self, event):
        event.Enable(True)

    def OnUpdateUIRedo(self, event):
        event.Enable(self.CanRedo())

    def OnUpdateUISave(self, event):
        event.Enable(self.CanSave())

    def OnUpdateUISaveAs(self, event):
        event.Enable(True)

    def OnUpdateUIUndo(self, event):
        event.Enable(self.CanUndo())

    def UpdateTitle(self):
        is_modified = not self.doc.saved

        self.OSXSetModified(is_modified)

        if self.doc.path:
            title = os.path.basename(self.doc.path)
            if is_modified:
                title = '* ' + title
        else:
            title = 'Unsaved'
        self.SetTitle(title)
