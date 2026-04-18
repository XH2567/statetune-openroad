# src/knowledge_rules.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

Condition = Dict[str, Any]
HardRule = Dict[str, Any]
SoftHeuristic = Dict[str, Any]

_VALID_OPS = {"==", "!=", "<", "<=", ">", ">=", "in", "not_in"}


def _ts_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _canon_scalar(v: Any) -> Any:
    """Normalize numeric-looking strings for stable keys."""
    if isinstance(v, str):
        s = v.strip()
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except Exception:
            return s
    return v


def _canon_value(v: Any) -> Any:
    """Normalize list/scalar for stable keys."""
    if isinstance(v, list):
        vv = [_canon_scalar(x) for x in v]
        try:
            return sorted(vv, key=lambda x: str(x))
        except Exception:
            return vv
    return _canon_scalar(v)


def condition_key(cond: Condition) -> str:
    """Canonical string key for a condition."""
    param = str(cond.get("param", "")).strip()
    op = str(cond.get("op", "")).strip()
    val = _canon_value(cond.get("value", None))
    return json.dumps(
        {"param": param, "op": op, "value": val},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def hard_rule_key(rule: HardRule) -> str:
    """Canonical key for hard rule dedup (order-insensitive for when/must)."""
    rule_type = str(rule.get("rule_type", "")).strip()
    when = rule.get("when", []) or []
    must = rule.get("must", []) or []
    when_keys = sorted([condition_key(c) for c in when if isinstance(c, dict)])
    must_keys = sorted([condition_key(c) for c in must if isinstance(c, dict)])
    payload = {
        "rule_type": rule_type,
        "when": when_keys,
        "must": must_keys,
        "message": str(rule.get("message", "")).strip(),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).casefold()


def soft_heuristic_key(h: SoftHeuristic) -> str:
    """Canonical key for soft heuristic dedup."""
    trigger = str(h.get("trigger", "")).strip()
    prefs = h.get("preferences", []) or []
    pref_keys = []
    for p in prefs:
        if not isinstance(p, dict):
            continue
        pref_keys.append(
            json.dumps(
                {
                    "param": str(p.get("param", "")).strip(),
                    "direction": str(p.get("direction", "")).strip(),
                    "strength": str(p.get("strength", "")).strip(),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    pref_keys.sort()
    payload = {"trigger": trigger, "preferences": pref_keys, "notes": str(h.get("notes", "")).strip()}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).casefold()


def eval_condition(params: Dict[str, Any], cond: Condition) -> bool:
    """Evaluate a single condition on params (AND semantics handled outside)."""
    if not isinstance(cond, dict):
        return False
    param = str(cond.get("param", "")).strip()
    op = str(cond.get("op", "")).strip()
    if (not param) or (op not in _VALID_OPS):
        return False

    left = params.get(param, None)
    right = cond.get("value", None)

    if op in ("in", "not_in"):
        if not isinstance(right, list):
            return False
        ok = left in right
        return ok if op == "in" else (not ok)

    def _to_num(x: Any) -> Optional[float]:
        try:
            if isinstance(x, bool):
                return float(int(x))
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                return float(x.strip())
        except Exception:
            return None
        return None

    ln = _to_num(left)
    rn = _to_num(right)

    # numeric comparisons
    if ln is not None and rn is not None and op in ("<", "<=", ">", ">="):
        if op == "<":
            return ln < rn
        if op == "<=":
            return ln <= rn
        if op == ">":
            return ln > rn
        if op == ">=":
            return ln >= rn
        return False

    # equality / inequality
    if ln is not None and rn is not None and op in ("==", "!="):
        ok = (ln == rn)
        return ok if op == "==" else (not ok)

    # fallback string compare
    ls = str(left).strip()
    rs = str(right).strip()
    ok = (ls == rs)
    return ok if op == "==" else (not ok)


def check_hard_rules(params: Dict[str, Any], hard_rules: List[HardRule]) -> List[str]:
    """
    Return violation messages (empty => pass).
    Semantics:
      - ban: if all 'when' true => violation
      - require: if all 'when' true AND not all 'must' true => violation
    """
    violations: List[str] = []
    for r in hard_rules or []:
        if not isinstance(r, dict):
            continue
        rt = str(r.get("rule_type", "")).strip()
        when = r.get("when", []) or []
        must = r.get("must", []) or []
        msg = str(r.get("message", "")).strip() or f"hard_rule_violation({rt})"
        if not when or not isinstance(when, list):
            continue

        if all(eval_condition(params, c) for c in when):
            if rt == "ban":
                violations.append(msg)
            elif rt == "require":
                if not must or (not all(eval_condition(params, c) for c in must)):
                    violations.append(msg)
    return violations


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def load_jsonl_unique(path: Path, *, key_func, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load unique items from JSONL, dedup by key_func(item), preserve recency.
    Returns oldest -> newest.
    """
    # Keep path normalization local so callers may pass str or Path safely.
    path = Path(path) if not isinstance(path, Path) else path

    items = _load_jsonl(path)
    if not items:
        return []
    seen = set()
    uniq_recent: List[Dict[str, Any]] = []
    for obj in reversed(items):
        try:
            k = key_func(obj)
        except Exception:
            continue
        if not k or k in seen:
            continue
        seen.add(k)
        uniq_recent.append(obj)
        if limit is not None and len(uniq_recent) >= int(limit):
            break
    return list(reversed(uniq_recent))


def append_jsonl_unique(path: Path, items: Iterable[Dict[str, Any]], *, key_func, max_scan: int = 800) -> None:
    """
    Append items into JSONL file with 'ts' if missing.
    Dedup by key_func, scanning tail of file for duplicates.
    """
    # Keep path normalization local so callers may pass str or Path safely.
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)

    recent_keys = set()
    if path.exists():
        raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = raw[-int(max_scan):] if len(raw) > int(max_scan) else raw
        for ln in tail:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            try:
                k = key_func(obj)
            except Exception:
                continue
            if k:
                recent_keys.add(k)

    with path.open("a", encoding="utf-8") as f:
        for obj in items:
            if not isinstance(obj, dict):
                continue
            obj2 = dict(obj)
            obj2.setdefault("ts", _ts_str())
            try:
                k = key_func(obj2)
            except Exception:
                continue
            if not k or k in recent_keys:
                continue
            f.write(json.dumps(obj2, ensure_ascii=False) + "\n")
            recent_keys.add(k)


def render_hard_rule_text(rule: HardRule) -> str:
    """Human-readable hard rule text for prompt injection / debugging."""
    rt = str(rule.get("rule_type", "")).strip()
    when = rule.get("when", []) or []
    must = rule.get("must", []) or []
    msg = str(rule.get("message", "")).strip()

    def _cond_txt(c: Condition) -> str:
        p = str(c.get("param", "")).strip()
        op = str(c.get("op", "")).strip()
        v = c.get("value", None)
        return f"{p} {op} {json.dumps(v, ensure_ascii=False)}"

    when_txt = " AND ".join(_cond_txt(c) for c in when if isinstance(c, dict))
    if rt == "ban":
        return f"BAN if ({when_txt}). {msg}".strip()
    if rt == "require":
        must_txt = " AND ".join(_cond_txt(c) for c in must if isinstance(c, dict))
        return f"REQUIRE: if ({when_txt}) then must satisfy ({must_txt}). {msg}".strip()
    return msg or f"HARD_RULE({rt}): {when_txt}"


def render_soft_heuristic_text(h: SoftHeuristic) -> str:
    """Human-readable soft heuristic text for prompt injection / debugging."""
    trig = str(h.get("trigger", "")).strip()
    prefs = h.get("preferences", []) or []
    notes = str(h.get("notes", "")).strip()

    parts = []
    for p in prefs:
        if not isinstance(p, dict):
            continue
        param = str(p.get("param", "")).strip()
        direction = str(p.get("direction", "")).strip()
        strength = str(p.get("strength", "")).strip()
        if strength:
            parts.append(f"{direction} {param} ({strength})")
        else:
            parts.append(f"{direction} {param}")
    pref_txt = "; ".join(parts)
    if trig:
        return f"[{trig}] {pref_txt}. {notes}".strip()
    return f"{pref_txt}. {notes}".strip()


def soft_preferences_from_heuristics(heuristics: List[SoftHeuristic]) -> Dict[str, int]:
    """
    Convert heuristics to simple param->sign preference:
      +1: increase/keep_high
      -1: decrease/keep_low
      0 : unknown/toggle/mixed
    """
    weight = {"": 1, "weak": 1, "medium": 2, "strong": 3}
    score: Dict[str, int] = {}
    for h in heuristics or []:
        prefs = h.get("preferences", []) or []
        for p in prefs:
            if not isinstance(p, dict):
                continue
            name = str(p.get("param", "")).strip()
            direction = str(p.get("direction", "")).strip().lower()
            strength = str(p.get("strength", "")).strip().lower()
            w = int(weight.get(strength, 1))
            s = 0
            if direction in ("increase", "keep_high"):
                s = +1
            elif direction in ("decrease", "keep_low"):
                s = -1
            else:
                s = 0
            if s == 0 or not name:
                continue
            score[name] = score.get(name, 0) + s * w

    out: Dict[str, int] = {}
    for k, v in score.items():
        out[k] = (+1 if v > 0 else -1 if v < 0 else 0)
    return out
