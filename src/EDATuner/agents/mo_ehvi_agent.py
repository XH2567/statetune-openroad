# src/mo_ehvi_agent.py
# Memory fix: lazy-load BoTorch/GPyTorch and release GP models proactively
import math
import random
import gc
from typing import Dict, List, Any, Optional

import torch

from .tuner_interface import TunerAgent

# ═══════════════════════════════════════════════════════════════════════════════
# Memory fix: lazy-load BoTorch/GPyTorch to avoid a ~1.5-2.5 GB startup cost
# ═══════════════════════════════════════════════════════════════════════════════

def _lazy_botorch():
    """Import BoTorch/GPyTorch on first use, cache the result, and reuse it later."""
    if not hasattr(_lazy_botorch, "_cache"):
        from botorch.models import SingleTaskGP, ModelListGP
        from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP  # ★ NEW
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.transforms.outcome import Standardize
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood  # ★ NEW
        _lazy_botorch._cache = {
            "SingleTaskGP": SingleTaskGP,
            "SingleTaskMultiFidelityGP": SingleTaskMultiFidelityGP,  # ★ NEW
            "ModelListGP": ModelListGP,
            "fit_gpytorch_mll": fit_gpytorch_mll,
            "Standardize": Standardize,
            "SobolQMCNormalSampler": SobolQMCNormalSampler,
            "qLogEHVI": qLogExpectedHypervolumeImprovement,
            "is_non_dominated": is_non_dominated,
            "Hypervolume": Hypervolume,
            "NondominatedPartitioning": NondominatedPartitioning,
            "SumMarginalLogLikelihood": SumMarginalLogLikelihood,
            "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,  # ★ NEW
        }
    return _lazy_botorch._cache


class MOEHVIAgent(TunerAgent):
    """
    Multi-objective surrogate + (q)EHVI agent (MVP):
    - objective direction: maximize ``[-area, wns, -power]``
    - acquisition: qEHVI with ``q=1`` (equivalent to EHVI)
    - cost-aware scoring: ``EHVI / E[time]`` with remaining-time gating
    """

    def __init__(self, param_space: List[Dict[str, Any]]):
        self.param_space = param_space
        self.param_names = [p["name"] for p in param_space]
        self.dim = len(param_space)
        self.bounds = self._get_bounds()  # raw bounds (2, d)
        self.dtypes = {p["name"]: p.get("type", "float") for p in param_space}

        self.warmup_trials: int = 0
        self.candidate_pool_size: int = 512
        self.mc_samples: int = 128

        # Memory guard: cap GP observations to avoid O(N^2) memory blow-up
        # N=350 → ~2.2GB peak, N=400 → ~2.9GB peak
        self.max_gp_obs: int = 350
        self._oom_retry_shrink: float = 0.75
        self._oom_max_retries: int = 3

        # Trust Region
        self.tr_center: Optional[torch.Tensor] = None  # (d,)
        self.tr_length: float = 1.0
        self.tr_min_length: float = 0.15
        self.tr_max_length: float = 1.0
        self.tr_shrink_factor: float = 0.85
        self.tr_grow_factor: float = 1.2

        self.best_hv: Optional[float] = None
        self.last_num_evals: int = 0

        # Quick-to-full calibration state
        self._calibration_pairs: List[Dict[str, Any]] = []  # [{params_fp, quick_m, full_m}]
        self._calibration_model: Optional[Any] = None  # sklearn LinearRegression
        self._calibration_r2: float = 0.0
        self._use_multi_fidelity: bool = True  # Whether to use MF-GP (can be disabled at runtime)

        # Promotion EHVI threshold
        self.min_ehvi_threshold: float = 0.0  # Computed dynamically; starts at 0
        self._recent_ehvi_values: List[float] = []  # Track recent EHVI values for an adaptive threshold
    # -------------------------------------------------------------------------
    # bounds / encode decode
    # -------------------------------------------------------------------------
    def _get_bounds(self) -> torch.Tensor:
        lower, upper = [], []
        for p in self.param_space:
            if "range" in p:
                lower.append(float(p["range"][0]))
                upper.append(float(p["range"][1]))
            elif "choices" in p:
                # choices may be numeric or string; we map to index space [0, len-1]
                lower.append(0.0)
                upper.append(float(max(0, len(p["choices"]) - 1)))
            else:
                lower.append(0.0)
                upper.append(1.0)
        return torch.tensor([lower, upper], dtype=torch.double)

    def _to_tensor(self, params: Dict[str, Any]) -> torch.Tensor:
        raw_vals = []
        for i, name in enumerate(self.param_names):
            p_cfg = self.param_space[i]
            v = params.get(name, 0.0)
            if "choices" in p_cfg:
                choices = list(p_cfg["choices"])
                try:
                    v = float(choices.index(v))
                except Exception:
                    v = 0.0
            raw_vals.append(v)
        t = torch.tensor([raw_vals], dtype=torch.double)
        lower = self.bounds[0]
        width = self.bounds[1] - self.bounds[0]
        width[width == 0] = 1.0
        t_norm = (t - lower) / width
        return t_norm.clamp(0.0, 1.0)

    def _to_dict(self, tensor_norm: torch.Tensor) -> Dict[str, Any]:
        lower = self.bounds[0]
        width = self.bounds[1] - self.bounds[0]
        t_raw = tensor_norm * width + lower
        t_raw = t_raw.squeeze().tolist()

        out: Dict[str, Any] = {}
        for i, name in enumerate(self.param_names):
            val = t_raw[i]
            p_cfg = self.param_space[i]

            if "choices" in p_cfg:
                choices = list(p_cfg["choices"])
                idx = int(round(val))
                idx = max(0, min(idx, len(choices) - 1))
                out[name] = choices[idx]
                continue

            if self.dtypes[name] in ["int", "categorical"]:
                val = int(round(val))
            else:
                val = float(val)
                val = round(val, 4)

            if "range" in p_cfg:
                mn, mx = p_cfg["range"]
                val = max(float(mn), min(float(val), float(mx)))
                if p_cfg.get("type") == "int":
                    val = int(val)

            out[name] = val
        return out

    def _random_sample(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for p in self.param_space:
            name = p["name"]
            if "range" in p:
                lo, hi = p["range"]
                if p.get("type") == "int":
                    params[name] = random.randint(int(lo), int(hi))
                else:
                    params[name] = random.uniform(float(lo), float(hi))
            elif "choices" in p:
                params[name] = random.choice(p["choices"])
            else:
                params[name] = 0.0
        return params

    # -------------------------------------------------------------------------
    # Memory guard: subsample intelligently to avoid O(N^2) GP fitting costs
    # -------------------------------------------------------------------------
    def _subsample_for_gp(
        self,
        valid: List[Dict[str, Any]],
        max_obs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        If valid history exceeds ``max_gp_obs``, subsample it to keep memory under control.

        Strategy:
          1) Keep all Pareto-front points (up to 40% of the budget)
          2) Keep the most recent 30% of records (to preserve exploration continuity)
          3) Randomly sample the remaining budget (to preserve coverage)
        """
        max_obs = max_obs or self.max_gp_obs
        if len(valid) <= max_obs:
            return valid

        n = len(valid)
        pareto_budget = int(max_obs * 0.40)
        recent_budget = int(max_obs * 0.30)

        # 1) Pareto-front points
        # Use BoTorch ``is_non_dominated`` for a fast Pareto filter
        try:
            B = _lazy_botorch()
            Y_all = torch.stack(
                [self._to_objectives_max(h["metrics"]) for h in valid], dim=0
            ).to(torch.double)
            nd_mask = B["is_non_dominated"](Y_all)
            pareto_indices = set(int(i) for i in torch.where(nd_mask)[0].tolist())
        except Exception:
            pareto_indices = set()

        # Truncate to the Pareto budget
        if len(pareto_indices) > pareto_budget:
            pareto_indices = set(random.sample(sorted(pareto_indices), pareto_budget))

        selected = set(pareto_indices)

        # 2) Most recent records
        recent_start = max(0, n - recent_budget)
        for i in range(recent_start, n):
            selected.add(i)

        # 3) Randomly sample the remaining budget
        remaining_budget = max(0, max_obs - len(selected))
        remaining_pool = [i for i in range(n) if i not in selected]
        if remaining_pool and remaining_budget > 0:
            sampled = random.sample(remaining_pool, min(len(remaining_pool), remaining_budget))
            selected.update(sampled)

        # Preserve the original order
        result = [valid[i] for i in sorted(selected)]
        print(f"  [EHVI] Memory guard: {n} → {len(result)} obs "
              f"(pareto={len(pareto_indices)}, recent={min(recent_budget, n - recent_start)}, "
              f"random={remaining_budget})")
        return result

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        """Check whether an exception looks like an out-of-memory error."""
        s = str(exc)
        return (
            "not enough memory" in s
            or "DefaultCPUAllocator" in s
            or "CUDA out of memory" in s
            or isinstance(exc, MemoryError)
        )

    def get_promising_regions(
        self,
        history: List[Dict[str, Any]],
        n_regions: int = 3,
        pool_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Use the EHVI acquisition function to find the most promising parameter regions.
        Return the top-N candidates as raw-space dictionaries for the KA prompt.
        Includes built-in OOM protection via automatic subsampling and retries.
        """
        valid = [h for h in history if self._is_valid_metrics(h.get("metrics", {}))]
        if len(valid) < max(6, self.warmup_trials):
            return []

        current_max = self.max_gp_obs
        for attempt in range(self._oom_max_retries + 1):
            sub_valid = self._subsample_for_gp(valid, max_obs=current_max)
            try:
                B = _lazy_botorch()
                X = torch.cat([self._to_tensor(h["params"]) for h in sub_valid], dim=0).to(torch.double)
                Y = torch.stack([self._to_objectives_max(h["metrics"]) for h in sub_valid], dim=0).to(torch.double)
                ref_point = self._choose_ref_point(Y)
                self._update_trust_region(X, Y, ref_point)
                model = self._fit_qor_model(X, Y)

                partitioning = B["NondominatedPartitioning"](ref_point=ref_point, Y=Y)
                sampler = B["SobolQMCNormalSampler"](sample_shape=torch.Size([128]))
                acq = B["qLogEHVI"](
                    model=model, ref_point=ref_point,
                    partitioning=partitioning, sampler=sampler,
                )

                cand_X = self._sample_candidate_pool(pool_size)
                model.eval()
                acq.eval()
                ehvi_vals = self._batch_eval_acq(acq, cand_X, batch_size=64)

                topk_idx = torch.topk(ehvi_vals, k=min(n_regions, len(ehvi_vals)), largest=True).indices
                regions = []
                for idx in topk_idx:
                    params = self._to_dict(cand_X[idx:idx+1])
                    regions.append({
                        "params": params,
                        "ehvi_score": round(float(ehvi_vals[idx].item()), 6),
                    })

                # Memory fix: release the GP model proactively
                del model, partitioning, sampler, acq
                gc.collect()

                return regions

            except (RuntimeError, MemoryError) as e:
                # OOM retry: shrink the observation budget
                if self._is_oom_error(e) and attempt < self._oom_max_retries:
                    old_max = current_max
                    current_max = max(50, int(current_max * self._oom_retry_shrink))
                    print(f"[EHVI][OOM] get_promising_regions attempt {attempt+1} failed "
                          f"(N={len(sub_valid)}). Shrinking {old_max} → {current_max}...")
                    gc.collect()
                    continue
                print(f"[EHVI] get_promising_regions failed: {e}")
                return []
            except Exception as e:
                print(f"[EHVI] get_promising_regions failed: {e}")
                return []

        return []


    # -------------------------------------------------------------------------
    # MO utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_valid_metrics(m: Dict[str, float]) -> bool:
        if not isinstance(m, dict):
            return False
        # Use sentinel flags instead of a hard-coded WNS threshold
        if any(m.get(f"__{k}_sentinel__", False) for k in ("area", "wns", "power")):
            return False
        area = float(m.get("area", float("inf")))
        wns = float(m.get("wns", -100.0))
        power = float(m.get("power", float("inf")))
        if (not math.isfinite(area)) or (not math.isfinite(power)) or (not math.isfinite(wns)):
            return False
        # Exclude only exact sentinel defaults rather than a ``<= -99`` heuristic
        if area == float("inf") or power == float("inf") or wns == -100.0:
            return False
        return True

    @staticmethod
    def _to_objectives_max(m: Dict[str, float]) -> torch.Tensor:
        # maximize [-area, wns, -power]
        area = float(m["area"])
        wns = float(m["wns"])
        power = float(m["power"])
        return torch.tensor([-area, wns, -power], dtype=torch.double)

    def _choose_ref_point(self, Y: torch.Tensor) -> torch.Tensor:
        y_min = Y.min(dim=0).values
        slack = 0.1 * Y.abs().mean(dim=0).clamp(min=1.0)
        return (y_min - slack).to(dtype=torch.double)

    def _update_trust_region(self, X: torch.Tensor, Y: torch.Tensor, ref_point: torch.Tensor) -> None:
        n = X.shape[0]
        if n <= 1 or n == self.last_num_evals:
            return

        B = _lazy_botorch()  # Memory fix: lazy import
        nd_mask = B["is_non_dominated"](Y)
        pareto_X = X[nd_mask]
        pareto_Y = Y[nd_mask]

        hv = float(B["Hypervolume"](ref_point).compute(pareto_Y))
        improved = (self.best_hv is None) or (hv > self.best_hv * (1.0 + 1e-6))
        if improved:
            self.best_hv = hv
            self.tr_length = min(self.tr_max_length, self.tr_length * self.tr_grow_factor)
        else:
            self.tr_length = max(self.tr_min_length, self.tr_length * self.tr_shrink_factor)

        # pick a center by simple scalarization on pareto
        y0 = pareto_Y
        rng = (y0.max(dim=0).values - y0.min(dim=0).values).clamp(min=1e-9)
        y01 = (y0 - y0.min(dim=0).values) / rng
        s = y01.sum(dim=1)
        best_idx = int(torch.argmax(s).item())
        self.tr_center = pareto_X[best_idx].clone().detach()

        # If there is no improvement for 20 consecutive updates, reset the trust region
        if not improved:
            _consecutive_no_improve = getattr(self, '_consecutive_no_improve', 0) + 1
            self._consecutive_no_improve = _consecutive_no_improve
            if _consecutive_no_improve >= 20:
                self.tr_length = self.tr_max_length
                self._consecutive_no_improve = 0
                print(f"[EHVI] Trust region RESET to {self.tr_max_length} after 20 consecutive non-improvements")
        else:
            self._consecutive_no_improve = 0

        self.last_num_evals = n


    def _sample_candidate_pool(self, n: int, *, global_fraction: float = 0.3) -> torch.Tensor:
        """Sample candidates with 70% inside the trust region and 30% globally."""
        if self.tr_center is None:
            self.tr_center = torch.full((self.dim,), 0.5, dtype=torch.double)

        n_global = max(1, int(n * global_fraction))
        n_local = n - n_global

        # Sample inside the trust region
        half = 0.5 * self.tr_length
        lo = (self.tr_center - half).clamp(0.0, 1.0)
        hi = (self.tr_center + half).clamp(0.0, 1.0)
        width = (hi - lo).clamp(min=1e-9)

        engine = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True)
        u_local = engine.draw(n_local).to(dtype=torch.double)
        X_local = lo + u_local * width

        # Global-space sampling
        engine_global = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True)
        X_global = engine_global.draw(n_global).to(dtype=torch.double)

        X = torch.cat([X_local, X_global], dim=0)
        return X.clamp(0.0, 1.0)


    def _fit_qor_model(self, X: torch.Tensor, Y: torch.Tensor):
        B = _lazy_botorch()  # Memory fix: lazy import
        models = []
        for j in range(Y.shape[1]):
            yj = Y[:, j : j + 1]
            gp = B["SingleTaskGP"](X, yj, outcome_transform=B["Standardize"](m=1))
            models.append(gp)
        model = B["ModelListGP"](*models)
        mll = B["SumMarginalLogLikelihood"](model.likelihood, model)
        B["fit_gpytorch_mll"](mll)
        return model
    # -------------------------------------------------------------------------
    # Quick-to-full calibration
    # -------------------------------------------------------------------------
    def record_calibration_pair(
        self,
        params_fp: str,
        quick_metrics: Dict[str, float],
        full_metrics: Dict[str, float],
    ) -> None:
        """Record a quick/full metric pair for calibration."""
        if not self._is_valid_metrics(quick_metrics) or not self._is_valid_metrics(full_metrics):
            return
        self._calibration_pairs.append({
            "params_fp": params_fp,
            "quick_area": float(quick_metrics.get("area", float("inf"))),
            "quick_wns": float(quick_metrics.get("wns", -100.0)),
            "quick_power": float(quick_metrics.get("power", float("inf"))),
            "full_area": float(full_metrics.get("area", float("inf"))),
            "full_wns": float(full_metrics.get("wns", -100.0)),
            "full_power": float(full_metrics.get("power", float("inf"))),
        })
        # Refit the calibration model every time 10+ pairs are available and the count hits a multiple of 5
        if len(self._calibration_pairs) >= 10 and len(self._calibration_pairs) % 5 == 0:
            self._fit_calibration_model()

    def _fit_calibration_model(self) -> None:
        """Fit a simple linear model to predict full metrics from quick metrics."""
        import numpy as np
        pairs = self._calibration_pairs
        if len(pairs) < 10:
            return

        X = np.array([[p["quick_area"], p["quick_wns"], p["quick_power"]] for p in pairs])
        Y = np.array([[p["full_area"], p["full_wns"], p["full_power"]] for p in pairs])

        # Check whether the quick metrics have enough variance
        x_std = X.std(axis=0)
        if np.any(x_std < 1e-12):
            print("[CALIB] Quick metrics have near-zero variance, calibration skipped")
            self._calibration_r2 = 0.0
            return

        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, Y)
            r2 = model.score(X, Y)
            self._calibration_model = model
            self._calibration_r2 = r2
            print(f"[CALIB] Calibration fitted: R²={r2:.3f} on {len(pairs)} pairs")

            if r2 < 0.3:
                print(f"[CALIB] WARNING: R²={r2:.3f} < 0.3, the quick-to-full proxy is highly distorted. "
                      f"Consider increasing QUICK_TARGET fidelity (for example, cts -> route_guide).")
        except ImportError:
            # Use NumPy least-squares when scikit-learn is unavailable
            import numpy as np
            X_aug = np.column_stack([X, np.ones(len(X))])
            try:
                coeffs, residuals, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
                Y_pred = X_aug @ coeffs
                ss_res = np.sum((Y - Y_pred) ** 2)
                ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
                self._calibration_model = ("numpy", coeffs)
                self._calibration_r2 = r2
                print(f"[CALIB] Calibration fitted (numpy): R²={r2:.3f} on {len(pairs)} pairs")
            except Exception as e:
                print(f"[CALIB] numpy lstsq failed: {e}")

    def predict_full_from_quick(
        self, quick_metrics: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """Predict full metrics with the calibration model. Return ``None`` if unavailable."""
        import numpy as np
        if self._calibration_model is None or self._calibration_r2 < 0.15:
            return None

        x = np.array([[
            float(quick_metrics.get("area", 0)),
            float(quick_metrics.get("wns", 0)),
            float(quick_metrics.get("power", 0)),
        ]])

        try:
            if isinstance(self._calibration_model, tuple) and self._calibration_model[0] == "numpy":
                coeffs = self._calibration_model[1]
                x_aug = np.column_stack([x, np.ones(1)])
                y = (x_aug @ coeffs).flatten()
            else:
                y = self._calibration_model.predict(x).flatten()
            return {"area": float(y[0]), "wns": float(y[1]), "power": float(y[2])}
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Multi-fidelity GP fitting
    # -------------------------------------------------------------------------
    def _fit_mf_qor_model(
        self,
        X_quick: torch.Tensor,
        Y_quick: torch.Tensor,
        X_full: torch.Tensor,
        Y_full: torch.Tensor,
    ):
        """
        Jointly model quick + full data with ``SingleTaskMultiFidelityGP``.
        The last column of ``X`` is the fidelity dimension (quick=0, full=1).
        Return a ``ModelListGP`` with one MF-GP per objective.
        """
        B = _lazy_botorch()
        n_q, d = X_quick.shape
        n_f = X_full.shape[0]

        # Append the fidelity column
        fid_q = torch.zeros(n_q, 1, dtype=torch.double)
        fid_f = torch.ones(n_f, 1, dtype=torch.double)

        X_all = torch.cat([
            torch.cat([X_quick, fid_q], dim=1),
            torch.cat([X_full, fid_f], dim=1),
        ], dim=0)

        Y_all = torch.cat([Y_quick, Y_full], dim=0)

        fidelity_dim = d  # The last column is fidelity

        models = []
        for j in range(Y_all.shape[1]):
            yj = Y_all[:, j:j+1]
            try:
                gp = B["SingleTaskMultiFidelityGP"](
                    X_all, yj,
                    data_fidelities=[fidelity_dim],
                    outcome_transform=B["Standardize"](m=1),
                )
                mll = B["ExactMarginalLogLikelihood"](gp.likelihood, gp)
                B["fit_gpytorch_mll"](mll)
                models.append(gp)
            except Exception as e:
                print(f"[MF-GP] Objective {j} MF fitting failed: {e}, fallback to SingleTaskGP")
                # Fallback: use full-fidelity data only
                gp = B["SingleTaskGP"](
                    X_full, Y_full[:, j:j+1],
                    outcome_transform=B["Standardize"](m=1),
                )
                models.append(gp)

        model = B["ModelListGP"](*models)
        # For MF models, append ``fidelity=1`` at prediction time
        return model, fidelity_dim

    def _predict_at_full_fidelity(
        self, model, cand_X: torch.Tensor, fidelity_dim: int
    ) -> torch.Tensor:
        """Evaluate an MF model at ``fidelity=1`` (full).
        Append a ``fidelity=1`` column to ``cand_X``."""
        n = cand_X.shape[0]
        fid_col = torch.ones(n, 1, dtype=torch.double)
        X_with_fid = torch.cat([cand_X, fid_col], dim=1)
        return X_with_fid
    
    def _batch_eval_acq(self, acq, cand_X: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Evaluate the acquisition function in batches to avoid peak memory spikes."""
        n = cand_X.shape[0]
        if n <= batch_size:
            with torch.no_grad():
                return acq(cand_X.unsqueeze(1)).view(-1)

        results = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = cand_X[start:end]
            with torch.no_grad():
                vals = acq(chunk.unsqueeze(1)).view(-1)
            results.append(vals)
        return torch.cat(results, dim=0)

    # -------------------------------------------------------------------------
    # NEW: score a given list (for promotion selection)
    # -------------------------------------------------------------------------
    def score_params_cost_aware(
        self,
        history: List[Dict[str, Any]],
        params_list: List[Dict[str, Any]],
        runtime_predictor: Any,
        *,
        remaining_sec: Optional[float] = None,
        margin_sec: float = 0.0,
        risk_beta: float = 2.0,
    ) -> List[Dict[str, Any]]:
        if not params_list:
            return []

        def _time_only_fallback():
            out = []
            for p in params_list:
                mt, st = runtime_predictor.predict(p)
                t_u = max(1.0, float(mt) + float(risk_beta) * float(st))
                t_u = min(t_u, 14400.0)   # Same 4h cap as above
                feasible = True
                if remaining_sec is not None:
                    feasible = (t_u <= max(0.0, float(remaining_sec) - float(margin_sec)))
                out.append({"params": p, "ehvi": 0.0, "time_ucb": t_u, "score": (0.0 if feasible else -1e30), "feasible": feasible})
            return out

        # if not enough full history, degrade to time-only scoring
        if len(history) < self.warmup_trials:
            return _time_only_fallback()

        valid = [h for h in history if self._is_valid_metrics(h.get("metrics", {}))]
        if len(valid) < 6:
            return _time_only_fallback()

        # Memory guard: subsampling + OOM retries
        current_max = self.max_gp_obs
        for attempt in range(self._oom_max_retries + 1):
            sub_valid = self._subsample_for_gp(valid, max_obs=current_max)

            X_full = torch.cat([self._to_tensor(h["params"]) for h in sub_valid], dim=0).to(dtype=torch.double)
            Y_full = torch.stack([self._to_objectives_max(h["metrics"]) for h in sub_valid], dim=0).to(dtype=torch.double)

            ref_point = self._choose_ref_point(Y_full)
            self._update_trust_region(X_full, Y_full, ref_point)

            model, partitioning, sampler, acq = None, None, None, None
            mf_fidelity_dim = None
            try:
                B = _lazy_botorch()

                # Try a multi-fidelity GP built from both quick and full data
                _used_mf = False
                if (self._use_multi_fidelity
                        and hasattr(self, '_quick_history_for_mf')
                        and len(self._quick_history_for_mf) >= 10):
                    try:
                        quick_valid = [h for h in self._quick_history_for_mf
                                       if self._is_valid_metrics(h.get("metrics", {}))]
                        if len(quick_valid) >= 10:
                            sub_quick = self._subsample_for_gp(quick_valid,
                                                               max_obs=min(current_max, len(quick_valid)))
                            X_q = torch.cat([self._to_tensor(h["params"]) for h in sub_quick],
                                            dim=0).to(dtype=torch.double)
                            Y_q = torch.stack([self._to_objectives_max(h["metrics"]) for h in sub_quick],
                                              dim=0).to(dtype=torch.double)
                            model, mf_fidelity_dim = self._fit_mf_qor_model(X_q, Y_q, X_full, Y_full)
                            _used_mf = True
                            print(f"[EHVI] Using Multi-Fidelity GP: {len(sub_quick)} quick + {len(sub_valid)} full")
                    except Exception as e:
                        print(f"[EHVI] MF-GP failed ({e}), falling back to standard GP")
                        model = None
                        _used_mf = False

                if model is None:
                    model = self._fit_qor_model(X_full, Y_full)

                partitioning = B["NondominatedPartitioning"](ref_point=ref_point, Y=Y_full)
                sampler = B["SobolQMCNormalSampler"](sample_shape=torch.Size([int(self.mc_samples)]))

                # For MF models, keep using the ref_point and full-fidelity partitioning for acquisition
                acq = B["qLogEHVI"](model=model, ref_point=ref_point,
                                    partitioning=partitioning, sampler=sampler)

                cand_X = torch.cat([self._to_tensor(p) for p in params_list], dim=0).to(dtype=torch.double)

                # MF models require an appended ``fidelity=1`` column
                if _used_mf and mf_fidelity_dim is not None:
                    cand_X = self._predict_at_full_fidelity(model, cand_X, mf_fidelity_dim)

                model.eval()
                acq.eval()
                ehvi = self._batch_eval_acq(acq, cand_X, batch_size=64)

                # Track EHVI values for an adaptive threshold
                ehvi_vals_float = [float(ehvi[i].item()) for i in range(len(params_list))]
                self._recent_ehvi_values.extend(ehvi_vals_float)
                # Keep only the most recent 500 values
                if len(self._recent_ehvi_values) > 500:
                    self._recent_ehvi_values = self._recent_ehvi_values[-500:]
                # Adaptive threshold: use the 25th percentile of recent positive EHVI values
                if len(self._recent_ehvi_values) >= 20:
                    import numpy as np
                    self.min_ehvi_threshold = float(
                        np.percentile([v for v in self._recent_ehvi_values if v > 0], 25)
                    ) if any(v > 0 for v in self._recent_ehvi_values) else 0.0

                out = []
                for i, p in enumerate(params_list):
                    mt, st = runtime_predictor.predict(p)
                    t_u = max(1.0, float(mt) + float(risk_beta) * float(st))
                    t_u = min(t_u, 14400.0)   # Cap at 4 hours

                    feasible = True
                    if remaining_sec is not None:
                        feasible = (t_u <= max(0.0, float(remaining_sec) - float(margin_sec)))

                    raw_ehvi = float(ehvi[i].item())

                    # Promotion filter: reject candidates below the EHVI threshold
                    if raw_ehvi < self.min_ehvi_threshold:
                        sc = -1e30  # Below the threshold, not worth promoting
                    elif feasible:
                        sc = raw_ehvi / t_u
                    else:
                        sc = -1e30

                    # Timing-awareness bonus
                    # If calibrated predictions exist, reward points whose WNS is close to 0
                    if hasattr(self, '_quick_metrics_cache') and feasible and sc > -1e20:
                        qm = self._quick_metrics_cache.get(id(p))
                        if qm is not None:
                            pred_full = self.predict_full_from_quick(qm)
                            if pred_full is not None and pred_full["wns"] > -0.02:
                                # Timing-near-feasible bonus: multiply by 1.5-3.0
                                wns_bonus = min(3.0, 1.5 + 10.0 * max(0, pred_full["wns"] + 0.02))
                                sc *= wns_bonus

                    out.append({
                        "params": p,
                        "ehvi": raw_ehvi,
                        "time_ucb": t_u,
                        "score": sc,
                        "feasible": feasible,
                        "above_threshold": raw_ehvi >= self.min_ehvi_threshold,
                    })

                # Memory fix: release GP models / acquisition objects to avoid memory buildup
                del model, partitioning, sampler, acq
                gc.collect()

                return out

            except (RuntimeError, MemoryError) as e:
                del model, partitioning, sampler, acq
                gc.collect()
                if self._is_oom_error(e) and attempt < self._oom_max_retries:
                    old_max = current_max
                    current_max = max(50, int(current_max * self._oom_retry_shrink))
                    print(f"[EHVI][OOM] score_params_cost_aware attempt {attempt+1} failed "
                          f"(N={len(sub_valid)}). Shrinking {old_max} → {current_max}...")
                    continue
                return _time_only_fallback()
            except Exception:
                del model, partitioning, sampler, acq
                gc.collect()
                return _time_only_fallback()

        return _time_only_fallback()

    # -------------------------------------------------------------------------
    # existing API: pick a brand-new point (for quick exploration)
    # -------------------------------------------------------------------------
    def suggest_params_cost_aware(
        self,
        history: List[Dict[str, Any]],
        runtime_predictor: Any,
        *,
        remaining_sec: Optional[float] = None,
        margin_sec: float = 0.0,
        risk_beta: float = 2.0,
        topk_eval: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if len(history) < self.warmup_trials:
            return self._random_sample()

        valid = [h for h in history if self._is_valid_metrics(h.get("metrics", {}))]
        if len(valid) < 6:
            return self._random_sample()

        # Memory guard: subsampling + OOM retries
        current_max = self.max_gp_obs
        for attempt in range(self._oom_max_retries + 1):
            sub_valid = self._subsample_for_gp(valid, max_obs=current_max)

            X = torch.cat([self._to_tensor(h["params"]) for h in sub_valid], dim=0).to(dtype=torch.double)
            Y = torch.stack([self._to_objectives_max(h["metrics"]) for h in sub_valid], dim=0).to(dtype=torch.double)

            ref_point = self._choose_ref_point(Y)
            self._update_trust_region(X, Y, ref_point)

            model, partitioning, sampler, acq = None, None, None, None
            try:
                B = _lazy_botorch()
                model = self._fit_qor_model(X, Y)
                partitioning = B["NondominatedPartitioning"](ref_point=ref_point, Y=Y)
                sampler = B["SobolQMCNormalSampler"](sample_shape=torch.Size([int(self.mc_samples)]))
                acq = B["qLogEHVI"](model=model, ref_point=ref_point, partitioning=partitioning, sampler=sampler)

                n_cand = int(topk_eval) if topk_eval is not None else int(self.candidate_pool_size)
                cand_X = self._sample_candidate_pool(n_cand)

                model.eval()
                acq.eval()
                ehvi = self._batch_eval_acq(acq, cand_X, batch_size=64)

                best_score = -1e30
                best_params: Optional[Dict[str, Any]] = None

                for i in range(n_cand):
                    cand_params = self._to_dict(cand_X[i : i + 1])
                    mean_t, std_t = runtime_predictor.predict(cand_params)
                    t_u = float(mean_t) + float(risk_beta) * float(std_t)
                    t_u = max(1.0, t_u)

                    if remaining_sec is not None:
                        if t_u > max(0.0, float(remaining_sec) - float(margin_sec)):
                            continue

                    score = float(ehvi[i].item()) / t_u
                    if score > best_score:
                        best_score = score
                        best_params = cand_params

                # Memory fix: release GP models / acquisition objects to avoid memory buildup
                del model, partitioning, sampler, acq
                gc.collect()

                return best_params

            except (RuntimeError, MemoryError) as e:
                del model, partitioning, sampler, acq
                gc.collect()
                if self._is_oom_error(e) and attempt < self._oom_max_retries:
                    old_max = current_max
                    current_max = max(50, int(current_max * self._oom_retry_shrink))
                    print(f"[EHVI][OOM] suggest_params_cost_aware attempt {attempt+1} failed "
                          f"(N={len(sub_valid)}). Shrinking {old_max} → {current_max}...")
                    continue
                return self._random_sample()
            except Exception:
                return self._random_sample()

        return self._random_sample()

    # -------------------------------------------------------------------------
    # TunerAgent interface stubs
    # -------------------------------------------------------------------------
    def suggest_params(self, history, *, runtime_predictor=None, remaining_sec=None, risk_beta=2.0, margin_sec=0.0):
        if len(history) < self.warmup_trials:
            return self._random_sample()
        if runtime_predictor is None:
            return self._random_sample()
        p = self.suggest_params_cost_aware(
            history,
            runtime_predictor=runtime_predictor,
            remaining_sec=remaining_sec,
            risk_beta=risk_beta,
            margin_sec=margin_sec,
        )
        return p or self._random_sample()


    def update(self, params: Dict, metrics: Dict):
        return
