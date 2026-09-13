"""Bounded Microduck desktop-arm control coordinator."""

from .contracts import DUAL_CONTRACT, SINGLE_CONTRACT
from .coordinator import Coordinator

__all__ = ["Coordinator", "SINGLE_CONTRACT", "DUAL_CONTRACT"]
