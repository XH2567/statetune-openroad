# src/runtime_model.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


@dataclass
class _Sample:
    x: np.ndarray
    y: float


class RuntimePredictor:
    """
    Lightweight kNN runtime surrogate.
    - observe(params, runtime_sec)
    - fit() -> bool  (True if enough samples)
    - predict(params) -> (mean, std)

    Compatible with old code that passes min_samples.
    """

    def __init__(
        self,
        param_space: List[Dict[str, Any]],
        *,
        min_samples: int = 10,
        k: int = 10,
        min_std: float = 1.0,
        prior_mean: float = 3600.0,
        prior_std: float = 1800.0,
    ):
        self.param_space = param_space
        self.min_samples = int(min_samples)
        self.k = int(k)
        self.min_std = float(min_std)
        self.prior_mean = float(prior_mean)
        self.prior_std = float(prior_std)

        self._samples: List[_Sample] = []
        self._ready: bool = False

    def _vec(self, params: Dict[str, Any]) -> np.ndarray:
        feats: List[float] = []
        for p in self.param_space:
            name = p["name"]
            v = params.get(name, p.get("default"))

            # numeric range
            if "range" in p:
                lo, hi = p["range"]
                lo = float(lo)
                hi = float(hi)
                vv = float(v)
                feats.append((vv - lo) / (hi - lo) if hi > lo else 0.0)
                continue

            # categorical/choices (numeric or string)
            if "choices" in p:
                choices = list(p["choices"])
                try:
                    idx = choices.index(v)
                except Exception:
                    idx = 0
                feats.append(idx / (len(choices) - 1) if len(choices) > 1 else 0.0)
                continue

            # fallback
            feats.append((hash(str(v)) % 997) / 997.0)

        return np.asarray(feats, dtype=float)

    def observe(self, params: Dict[str, Any], runtime_sec: Any) -> None:
        try:
            y = float(runtime_sec)
        except Exception:
            return
        if (not math.isfinite(y)) or y <= 0:
            return
        self._samples.append(_Sample(self._vec(params), y))

    def fit(self) -> bool:
        self._ready = (len(self._samples) >= self.min_samples)
        return self._ready

    def num_samples(self) -> int:
        return len(self._samples)

    def predict(self, params: Dict[str, Any]) -> Tuple[float, float]:
        # no samples
        if not self._samples:
            return float(self.prior_mean), float(self.prior_std)

        # if not ready, still give empirical mean/std as weak estimate
        if not self._ready:
            y = np.asarray([s.y for s in self._samples], dtype=float)
            mean = float(np.mean(y))
            std = float(np.std(y) if len(y) > 1 else self.prior_std)
            return mean, max(self.min_std, std)

        x = self._vec(params)
        X = np.stack([s.x for s in self._samples], axis=0)
        y = np.asarray([s.y for s in self._samples], dtype=float)

        d = np.linalg.norm(X - x[None, :], axis=1)
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        d_k = d[idx]
        y_k = y[idx]

        w = 1.0 / (d_k + 1e-6)
        w = w / (w.sum() + 1e-12)

        mean = float(np.sum(w * y_k))
        var = float(np.sum(w * (y_k - mean) ** 2))
        std = float(max(self.min_std, math.sqrt(max(var, 0.0))))
        return mean, std

    def ucb(self, params: Dict[str, Any], beta: float = 2.0) -> float:
        m, s = self.predict(params)
        return float(m + beta * s)


class MultiFidelityRuntimePredictor:
    """
    Separate runtime predictors for quick/full.
    ★ FIX: 当 full 无数据时，用 quick 实测 × ratio 来估计，
    避免盲目 prior 导致 promote 被时间预算挡死。
    """

    FULL_QUICK_RATIO = 3.0          # full 通常是 quick 的 2-4 倍
    BOOTSTRAP_RISK_BETA = 1.0       # bootstrap 阶段用更低的 risk_beta

    def __init__(
        self,
        param_space: List[Dict[str, Any]],
        *,
        min_samples_quick: int = 10,
        min_samples_full: int = 10,
    ):
        self.quick = RuntimePredictor(
            param_space,
            min_samples=min_samples_quick,
            prior_mean=1200.0,
            prior_std=900.0,
        )
        self.full = RuntimePredictor(
            param_space,
            min_samples=min_samples_full,
            prior_mean=3600.0,
            prior_std=1200.0,
        )

    def observe(self, fidelity: str, params: Dict[str, Any], runtime_sec: Any) -> None:
        if fidelity == "quick":
            self.quick.observe(params, runtime_sec)
        else:
            self.full.observe(params, runtime_sec)

    def fit(self, fidelity: Optional[str] = None) -> None:
        if fidelity is None:
            self.quick.fit()
            self.full.fit()
        elif fidelity == "quick":
            self.quick.fit()
        else:
            self.full.fit()

    def predict_full_bootstrapped(self, params: Dict[str, Any]) -> Tuple[float, float]:
        """
        ★ NEW: 预测 full runtime，优先用 full 实测数据；
        无 full 数据时用 quick 实测 × ratio 引导。
        """
        # 有 full 观测 → 直接用
        if self.full._samples:
            return self.full.predict(params)

        # 无 full 但有 quick 观测 → bootstrap
        if self.quick._samples:
            q_mean, q_std = self.quick.predict(params)
            return (
                q_mean * self.FULL_QUICK_RATIO,
                q_std * self.FULL_QUICK_RATIO,
            )

        # 都没有 → 用 full prior
        return self.full.prior_mean, self.full.prior_std
    




class IterationBudgetEstimator:
    """
    Innovus-only 模式的预算估算器。
    不再依赖 wall-clock deadline，而是用「剩余迭代数 × 预估单次时间」。

    用法:
        ibe = IterationBudgetEstimator(est_sec=1800, max_concurrency=32)
        ibe.remaining_sec(remaining_iters=50)      # → 90000.0
        ibe.budget_remaining(remaining_iters=50)    # → 2880000.0 (thread-seconds)
    """

    def __init__(self, estimated_sec_per_iter: float, max_concurrency: int = 1):
        self.est_sec = float(max(1.0, estimated_sec_per_iter))
        self.max_concurrency = int(max(1, max_concurrency))
        # 运行时自适应: 用实际观测值逐渐修正 est_sec
        self._observed: List[float] = []
        self._use_observed: bool = False

    def observe(self, actual_sec: float) -> None:
        """记录实际 Innovus 运行时间，用于自适应修正预估。"""
        if actual_sec > 0 and math.isfinite(actual_sec):
            self._observed.append(actual_sec)
            # 积累足够样本后切换到实测均值
            if len(self._observed) >= 5:
                self._use_observed = True

    @property
    def effective_est_sec(self) -> float:
        """返回当前有效的单次预估时间。有足够实测数据时用实测均值，否则用配置值。"""
        if self._use_observed and self._observed:
            return float(np.mean(self._observed[-50:]))  # 滑动窗口最近 50 次
        return self.est_sec

    def remaining_sec(self, remaining_iters: int) -> float:
        """估算剩余 wall-clock 时间（单线程视角）。"""
        return max(0.0, remaining_iters * self.effective_est_sec)

    def budget_remaining(self, remaining_iters: int) -> float:
        """估算剩余算力预算（thread-seconds），与原 compute_budget_remaining 语义一致。"""
        return max(0.0, remaining_iters * self.effective_est_sec * self.max_concurrency)
