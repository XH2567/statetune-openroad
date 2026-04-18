# src/candidate.py
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class TclOverride(BaseModel):
    model_config = ConfigDict(extra="allow")
    hook_var: str = "FASTROUTE_TCL"
    content: str


class Candidate(BaseModel):
    """
    Candidate 一等公民（Phase3）：
      - params: 结构化参数
      - fidelity: quick/full（多保真）
      - parent_run_id: quick -> full 的升格链
      - tcl_overrides: 可选 Tcl hook 注入
    """
    model_config = ConfigDict(extra="allow")

    params: Dict[str, Any] = Field(default_factory=dict)
    tcl_overrides: List[TclOverride] = Field(default_factory=list)

    fidelity: Literal["quick", "full"] = "full"
    parent_run_id: Optional[str] = None

    origin: str = "rule"
    rationale: Optional[str] = None
    created_ts: float = Field(default_factory=lambda: float(time.time()))

    def flat_make_vars(self) -> Dict[str, Any]:
        return dict(self.params)

    def to_script_override(self) -> Optional[Dict[str, Any]]:
        if not self.tcl_overrides:
            return None
        t0 = self.tcl_overrides[0]
        return {"hook_var": t0.hook_var, "content": t0.content}

    def fingerprint(self) -> str:
        """
        用于去重：只基于 params+tcl_overrides（不包含 fidelity / parent）
        """
        obj = {
            "params": self._stable_params(self.params),
            "tcl_overrides": [
                {"hook_var": (t.hook_var or "").strip().upper(), "content": t.content or ""}
                for t in (self.tcl_overrides or [])
            ],
        }
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _stable_params(params: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, float):
                out[k] = round(v, 6)
            else:
                out[k] = v
        return out
