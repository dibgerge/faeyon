"""
Global mode manager for faeyon library.

This module provides a global mode that controls how Delayable objects behave
when data is fed to them via the >> operator.

Modes:
- "eager": Direct calculation mode - data >> Delayable() immediately resolves
- "lazy": Tracing mode - data >> Delayable() caches the data and returns a new Delayable
"""
from __future__ import annotations
from typing import Literal
from contextlib import contextmanager
from ..utils import Singleton


modeType = Literal["eager", "lazy"]


class _FaeMode(metaclass=Singleton):
    """
    Global mode manager for faeyon library.
    
    Controls whether Delayable objects should:
    - "eager": Immediately resolve when data is provided
    - "lazy": Cache data and return a new Delayable for graph tracing
    
    Usage:
        # Set mode globally
        faeyon_mode.set("lazy")
        
        # Use as context manager
        with faeyon_mode("lazy"):
            # Code in lazy mode
    """
    def __init__(self):
        self._mode: modeType = "eager"
    
    @property
    def mode(self) -> modeType:
        """Get the current mode."""
        return self._mode

    @property
    def is_eager(self) -> bool:
        """Check if currently in eager mode."""
        return self._mode == "eager"
    
    @property
    def is_lazy(self) -> bool:
        """Check if currently in lazy mode."""
        return self._mode == "lazy"

    @mode.setter
    def mode(self, mode: modeType) -> None:
        """Set the current mode."""
        self.set(mode)
    
    def set(self, mode: modeType) -> None:
        """
        Set the global mode.
        
        Args:
            mode: Either "eager" or "lazy"
        """
        if mode not in ("eager", "lazy"):
            raise ValueError(f"Mode must be 'eager' or 'lazy', got {mode}")
        self._mode = mode

    @contextmanager
    def __call__(self, mode: modeType) -> None:
        old_mode = self._mode

        with self._lock:
            self._mode = mode
            try:
                yield
            finally:
                self._mode = old_mode


# Global singleton instance
fae_mode = _FaeMode()
