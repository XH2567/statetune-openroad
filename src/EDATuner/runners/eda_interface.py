# src/eda_interface.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class EDARunner(ABC):
    """
    Abstract base class for an EDA flow runner.

    A runner should:
      1) materialize a run directory for each run_id
      2) execute the underlying tool flow (OpenROAD-flow-scripts, Cadence Genus/Innovus, etc.)
      3) write a metrics.json in the run directory
      4) provide get_metrics(log_path) to load metrics for downstream ranking

    Notes:
      - `script_override` is runner-specific. For some flows it may be a Tcl snippet
        that will be sourced, or a structured object describing how to override scripts.
      - Additional kwargs (e.g., make_target, timeout_sec) are accepted for compatibility
        with existing multi-fidelity scheduling code.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    @abstractmethod
    def run(
        self,
        params: Dict[str, Any],
        run_id: str,
        script_override: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """
        Run one EDA experiment.

        Args:
            params: Tunable parameters (from BO / LLM).
            run_id: Experiment ID (used as directory name).
            script_override: Optional runner-specific override payload.
            **kwargs: Optional runner-specific controls (e.g., make_target, timeout_sec).

        Returns:
            (ok, log_path)
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, log_path: str) -> Dict[str, Any]:
        """
        Load PPA metrics for a run (typically from run_dir/metrics.json).

        Returns:
            Dict with at least:
              - area (float, lower is better)
              - wns  (float, higher is better)
              - power (float, lower is better)
        """
        raise NotImplementedError

    def cleanup(self, history: List[Dict[str, Any]]) -> None:
        """Optional cleanup hook called periodically by main loop."""
        return
