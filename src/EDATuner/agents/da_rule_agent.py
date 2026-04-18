# src/da_rule_agent.py
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ..optimization.candidate import Candidate


def _is_valid_metrics(m: Dict[str, float]) -> bool:
    try:
        area = float(m.get("area", float("inf")))
        wns = float(m.get("wns", -100.0))
        power = float(m.get("power", float("inf")))
        if not (math.isfinite(area) and math.isfinite(wns) and math.isfinite(power)):
            return False
        if area == float("inf") or power == float("inf") or wns <= -99.0:
            return False
        return True
    except Exception:
        return False


def _pareto_front_ids(valid_hist: List[Dict[str, Any]]) -> List[int]:
    """
    Objectives: Area (min), WNS (max), Power (min).
    Return the indices of non-dominated points in valid_hist.
    """
    ids: List[int] = []
    for i, hi in enumerate(valid_hist):
        yi = hi["metrics"]
        dominated = False
        for j, hj in enumerate(valid_hist):
            if i == j:
                continue
            yj = hj["metrics"]
            # yj dominates yi if: area<=, power<=, wns>= and at least one strict
            if (
                float(yj["area"]) <= float(yi["area"])
                and float(yj["power"]) <= float(yi["power"])
                and float(yj["wns"]) >= float(yi["wns"])
                and (
                    float(yj["area"]) < float(yi["area"])
                    or float(yj["power"]) < float(yi["power"])
                    or float(yj["wns"]) > float(yi["wns"])
                )
            ):
                dominated = True
                break
        if not dominated:
            ids.append(i)
    return ids


class RuleBasedDA:
    """
    Phase 2.5: an interpretable DA module without LLMs
    - Learn coarse search directions from Pareto points and history (parameter vs. utility correlation).
    - Generate a mixture of candidates: directed local perturbation / local random / global random.
    - Attach a rationale to each candidate (can be logged to the KB).
    """

    def __init__(
        self,
        param_space: List[Dict[str, Any]],
        *,
        random_state: int = 0,
        local_step_frac: float = 0.08,
        local_mutate_k: int = 2,
        mix_local_directed: float = 0.55,
        mix_local_random: float = 0.30,
        mix_global_random: float = 0.15,
    ):
        self.param_space = param_space
        self.rng = random.Random(int(random_state))

        self.local_step_frac = float(local_step_frac)
        self.local_mutate_k = int(local_mutate_k)

        # Mixture weights (automatically normalized)
        s = float(mix_local_directed + mix_local_random + mix_global_random)
        self.w_directed = mix_local_directed / s
        self.w_local_rand = mix_local_random / s
        self.w_global = mix_global_random / s

    def generate(
        self,
        history: List[Dict[str, Any]],
        *,
        n: int = 64,
        soft_preferences: Optional[Dict[str, int]] = None,
    ) -> List[Candidate]:
        valid = [h for h in history if _is_valid_metrics(h.get("metrics", {})) and isinstance(h.get("params", None), dict)]
        pareto_idxs: List[int] = []
        if valid:
            pareto_idxs = _pareto_front_ids(valid)

        corr = self._estimate_param_utility_corr(valid)

        out: List[Candidate] = []
        for _ in range(int(n)):
            r = self.rng.random()

            if (not valid) or (r < self.w_global):
                params = self._sample_global()
                rationale = "global random exploration (no/low history)"
                parent = None
            else:
                base = valid[self.rng.choice(pareto_idxs)] if pareto_idxs else self.rng.choice(valid)
                base_params = dict(base["params"])
                parent = str(base.get("run_id", None))

                if r < self.w_global + self.w_local_rand:
                    params = self._mutate_local_random(base_params)
                    rationale = f"local random perturbation around {parent}"
                else:
                    params = self._mutate_local_directed(base_params, corr, soft_preferences=soft_preferences)
                    rationale = f"directed mutation around {parent} using corr + soft_preferences"

            out.append(
                Candidate(
                    params=params,
                    origin="rule",
                    rationale=rationale,
                    parent_run_id=parent,
                )
            )
        return out

    def _mutate_local_directed(
        self,
        base: Dict[str, Any],
        corr: Dict[str, float],
        *,
        soft_preferences: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        params = dict(base)
        pref = soft_preferences or {}

        # score = |corr| + small bonus if soft pref exists
        scored = []
        for p in self.param_space:
            name = p["name"]
            c = float(corr.get(name, 0.0)) if isinstance(corr, dict) else 0.0
            bonus = 0.25 if (name in pref and int(pref.get(name, 0)) != 0) else 0.0
            imp = abs(c) + bonus

            # direction: soft pref overrides corr sign if provided
            sign = (1 if c >= 0 else -1)
            if name in pref and int(pref.get(name, 0)) != 0:
                sign = 1 if int(pref[name]) > 0 else -1

            scored.append((imp, name, sign))

        scored.sort(reverse=True, key=lambda t: t[0])

        if not scored:
            return self._mutate_local_random(base)

        k = max(1, min(self.local_mutate_k, len(scored)))
        for _, name, sign in scored[:k]:
            params[name] = self._nudge_one(name, params.get(name), directed_sign=sign)

        return params

    # -------------------------
    # Corr estimation
    # -------------------------
    def _estimate_param_utility_corr(self, valid: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        utility = z(-area) + z(wns) + z(-power)
        corr(param, utility): sign indicates direction and abs value indicates importance
        """
        if len(valid) < 6:
            return {}

        # utilities
        vals_u: List[float] = []
        ys_area: List[float] = []
        ys_wns: List[float] = []
        ys_power: List[float] = []
        for h in valid:
            m = h["metrics"]
            ys_area.append(-float(m["area"]))
            ys_wns.append(float(m["wns"]))
            ys_power.append(-float(m["power"]))

        def zscore(xs: List[float]) -> List[float]:
            mu = sum(xs) / len(xs)
            var = sum((x - mu) ** 2 for x in xs) / max(1, (len(xs) - 1))
            sd = math.sqrt(var) if var > 1e-12 else 1.0
            return [(x - mu) / sd for x in xs]

        za, zw, zp = zscore(ys_area), zscore(ys_wns), zscore(ys_power)
        for i in range(len(valid)):
            vals_u.append(za[i] + zw[i] + zp[i])

        # corr per param
        corr: Dict[str, float] = {}
        for p in self.param_space:
            name = p["name"]
            xs: List[float] = []
            for h in valid:
                v = h["params"].get(name, None)
                if v is None:
                    continue
                if isinstance(v, bool):
                    xs.append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)):
                    xs.append(float(v))
                else:
                    # non-numeric -> skip corr
                    xs = []
                    break

            if len(xs) != len(vals_u) or len(xs) < 6:
                continue

            c = self._pearson(xs, vals_u)
            if math.isfinite(c):
                corr[name] = float(c)

        return corr

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> float:
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        mx = sum(x[:n]) / n
        my = sum(y[:n]) / n
        vx = sum((xi - mx) ** 2 for xi in x[:n]) / max(1, (n - 1))
        vy = sum((yi - my) ** 2 for yi in y[:n]) / max(1, (n - 1))
        sx = math.sqrt(vx) if vx > 1e-12 else 1.0
        sy = math.sqrt(vy) if vy > 1e-12 else 1.0
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / max(1, (n - 1))
        return cov / (sx * sy)

    # -------------------------
    # Sampling / mutation
    # -------------------------
    def _sample_global(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for p in self.param_space:
            name = p["name"]
            if "choices" in p:
                params[name] = self.rng.choice(list(p["choices"]))
            elif "range" in p:
                lo, hi = p["range"]
                if p.get("type") == "int":
                    params[name] = int(self.rng.randint(int(lo), int(hi)))
                else:
                    params[name] = round(self.rng.uniform(float(lo), float(hi)), 6)
            else:
                params[name] = 0
        return params

    def _mutate_local_random(self, base: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(base)
        # Randomly choose k parameters and apply small perturbations
        names = [p["name"] for p in self.param_space]
        self.rng.shuffle(names)
        k = max(1, min(self.local_mutate_k, len(names)))
        for name in names[:k]:
            params[name] = self._nudge_one(name, params.get(name), directed_sign=None)
        return params

    def _mutate_local_directed(self, base: Dict[str, Any], corr: Dict[str, float]) -> Dict[str, Any]:
        params = dict(base)

        # Sort by |corr| and prioritize more informative parameters
        scored = []
        for p in self.param_space:
            name = p["name"]
            if name in corr:
                scored.append((abs(corr[name]), name, corr[name]))
        scored.sort(reverse=True, key=lambda t: t[0])

        if not scored:
            return self._mutate_local_random(base)

        k = max(1, min(self.local_mutate_k, len(scored)))
        for _, name, c in scored[:k]:
            sign = 1 if c >= 0 else -1
            params[name] = self._nudge_one(name, params.get(name), directed_sign=sign)
        return params

    def _nudge_one(self, name: str, cur: Any, directed_sign: Optional[int]) -> Any:
        p = next((pp for pp in self.param_space if pp["name"] == name), None)
        if p is None:
            return cur

        if "choices" in p:
            ch = list(p["choices"])
            if not ch:
                return cur
            # For binary choices, flip according to the sign; otherwise sample randomly
            if len(ch) == 2 and directed_sign is not None:
                return ch[1] if (cur == ch[0]) else ch[0]
            return self.rng.choice(ch)

        if "range" not in p:
            return cur

        lo, hi = float(p["range"][0]), float(p["range"][1])
        width = (hi - lo) if (hi - lo) != 0 else 1.0
        step = self.local_step_frac * width

        if p.get("type") == "int":
            step_i = max(1, int(round(step)))
            try:
                x = int(round(float(cur)))
            except Exception:
                x = int(round((lo + hi) / 2))
            s = directed_sign if directed_sign is not None else (1 if self.rng.random() < 0.5 else -1)
            x2 = x + s * step_i
            x2 = int(max(lo, min(x2, hi)))
            return x2

        # float
        try:
            x = float(cur)
        except Exception:
            x = (lo + hi) / 2
        s = directed_sign if directed_sign is not None else (1 if self.rng.random() < 0.5 else -1)
        x2 = x + s * step * self.rng.uniform(0.5, 1.5)
        x2 = max(lo, min(x2, hi))
        return round(float(x2), 6)
