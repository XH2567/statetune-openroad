from __future__ import annotations

from typing import Any, Dict

from .eda_interface import EDARunner


def create_runner(cfg: Dict[str, Any]) -> EDARunner:
    """
    Create an EDARunner instance from the provided config.

    Supported values:
      - eda.flow: "openroad" / "orfs" -> OpenRoadFlow
      - eda.flow: "cadence"           -> CadenceFlow

    Auto-detection:
      - if cfg contains a top-level "cadence" section -> CadenceFlow
      - otherwise -> OpenRoadFlow
    """
    eda_cfg = cfg.get("eda", {}) or {}
    flow = str(eda_cfg.get("flow") or cfg.get("flow") or "").strip().lower()

    if not flow:
        flow = "cadence" if isinstance(cfg.get("cadence", None), dict) else "openroad"

    if flow in ("cadence", "innovus", "genus"):
        from .cadence_runner import CadenceFlow

        return CadenceFlow(cfg)

    # Default / fallback.
    from .openroad_runner import OpenRoadFlow

    return OpenRoadFlow(cfg)
