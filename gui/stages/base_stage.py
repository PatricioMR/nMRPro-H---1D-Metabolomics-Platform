# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/stages/base_stage.py
"""
Base class for all analysis stages.
"""
from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import ttk


class BaseStage(ABC):
    """
    Abstract base class for all analysis stages.
    Each stage must implement create_widgets and on_activate.
    """

    def __init__(self, parent, app):
        """
        Initialize the stage.

        Args:
            parent: parent widget (left panel)
            app: reference to the main App instance
        """
        self.parent = parent
        self.app = app
        self.frame = None

    @abstractmethod
    def create_widgets(self):
        """
        Create all widgets for this stage.
        Must be implemented by each stage.
        """
        pass

    @abstractmethod
    def on_activate(self):
        """
        Called when this stage becomes active.
        Must be implemented by each stage.
        """
        pass

    def on_deactivate(self):
        """
        Called when this stage is deactivated.
        Default implementation hides the frame.
        """
        if self.frame:
            self.frame.pack_forget()

