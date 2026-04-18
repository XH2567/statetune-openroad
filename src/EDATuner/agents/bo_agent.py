# bo_agent.py
import torch
import random
import numpy as np
import math 
from typing import Dict, List, Any, Optional

# Core BoTorch imports
from botorch.models import PairwiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

from .tuner_interface import TunerAgent
from ..utils.ranktuner_utils import generate_comparisons


class RankTunerAgent(TunerAgent):
    """
    RankTuner-style agent implementation (keeping the original interface):
    - The constructor only accepts param_space: List[Dict].
    - Uses PairwiseGP + PairwiseLaplaceMarginalLogLikelihood to learn a preference function f(x).
    - Implements a Duel-Thompson style workflow:
        1) winner: sample from the posterior and choose the point with the highest average win rate
        2) challenger: with the winner fixed, choose the point maximizing Var[f(winner) - f(x)]
       The outer loop currently evaluates only one point, so this method returns the challenger (maximum information gain).
    - Includes a simple trust region:
        - Maintains a center and side length in [0,1]^d to control the candidate sampling range.
    """

    def __init__(self, param_space: List[Dict]):
        # Keep the original interface unchanged
        self.param_space = param_space
        self.param_names = [p["name"] for p in param_space]
        self.dim = len(param_space)
        self.bounds = self._get_bounds()  # Lower and upper bounds in the original space
        self.dtypes = {p["name"]: p.get("type", "float") for p in param_space}

        # Cached GP model
        self.model: Optional[PairwiseGP] = None

        # ---- Internal hyperparameters (kept here as defaults instead of __init__ arguments) ----
        self.warmup_trials: int = 5              # First N trials are purely random
        self.candidate_pool_size: int = 256      # Candidate pool size per iteration (Sobol sampling inside the trust region)
        self.ts_num_samples: int = 16            # Number of posterior samples used when selecting the winner

        # Trust region parameters in the normalized [0,1]^d space
        self.tr_center: Optional[torch.Tensor] = None  # shape: (d,)
        self.tr_length: float = 1.0
        self.tr_min_length: float = 0.05
        self.tr_max_length: float = 1.0
        self.tr_shrink_factor: float = 0.7
        self.tr_grow_factor: float = 1.3

        # Simple scalar score used to detect improvement (sum over objectives for now)
        self.best_score: Optional[float] = None
        self.last_num_evals: int = 0  # Number of samples seen in the previous update

    # -------------------------------------------------------------------------
    # Basic utilities: bounds, encoding, and decoding
    # -------------------------------------------------------------------------

    def _get_bounds(self) -> torch.Tensor:
        lower = []
        upper = []
        for p in self.param_space:
            if "range" in p:
                lower.append(p["range"][0])
                upper.append(p["range"][1])
            elif "choices" in p:
                lower.append(min(p["choices"]))
                upper.append(max(p["choices"]))
            else:
                lower.append(0.0)
                upper.append(1.0)
        return torch.tensor([lower, upper], dtype=torch.double)

    def _to_tensor(self, params: Dict) -> torch.Tensor:
        """Convert a parameter dict to a normalized [0,1] tensor with shape (1, d)."""
        raw_vals = []
        for name in self.param_names:
            raw_vals.append(params.get(name, 0.0))
        t = torch.tensor([raw_vals], dtype=torch.double)
        lower = self.bounds[0]
        range_width = self.bounds[1] - self.bounds[0]
        range_width[range_width == 0] = 1.0
        t_norm = (t - lower) / range_width
        return t_norm.clamp(0.0, 1.0)

    def _to_dict(self, tensor_norm: torch.Tensor) -> Dict:
        """Convert a normalized tensor back to a parameter dictionary."""
        lower = self.bounds[0]
        range_width = self.bounds[1] - self.bounds[0]
        t_raw = tensor_norm * range_width + lower
        t_raw = t_raw.squeeze().tolist()
        params = {}
        for i, name in enumerate(self.param_names):
            val = t_raw[i]
            p_cfg = self.param_space[i]

            if self.dtypes[name] in ["int", "categorical"]:
                val = int(round(val))
            else:
                val = float(val)
                val = round(val, 4)

            # Clamp to bounds / snap to the nearest choice
            if "range" in p_cfg:
                min_v, max_v = p_cfg["range"]
                val = max(min_v, min(val, max_v))
                if p_cfg.get("type") == "int":
                    val = int(val)
            elif "choices" in p_cfg:
                choices = p_cfg["choices"]
                # Assume choices are numeric
                val = min(choices, key=lambda x: abs(x - val))

            params[name] = val
        return params

    # -------------------------------------------------------------------------
    # Trust region helpers
    # -------------------------------------------------------------------------

    def _update_trust_region(self, valid_history: List[Dict]):
        """
        Simplified trust-region update strategy:
        - Center the region around the current best point
        - If the current best improves on the historical best, expand the region
        - Otherwise, shrink the region
        """
        if not valid_history:
            return

        n = len(valid_history)
        if n <= 1:
            return

        # Update the trust region only when new samples arrive
        if n == self.last_num_evals:
            return

        # Simple multi-objective aggregation: smaller score is better (delay + power + area)
        def _score(m):
            metrics = m.get("metrics", {})
            delay = metrics.get("delay", 0.0)
            power = metrics.get("power", 0.0)
            area = metrics.get("area", 0.0)
            return delay + power + area

        best_entry = min(valid_history, key=_score)
        cur_best_score = _score(best_entry)
        x_best_norm = self._to_tensor(best_entry["params"]).squeeze(0)

        if self.best_score is None:
            self.best_score = cur_best_score
            self.tr_center = x_best_norm
            self.last_num_evals = n
            return

        improved = cur_best_score < self.best_score - 1e-9

        if improved:
            self.best_score = cur_best_score
            self.tr_center = x_best_norm
            self.tr_length = min(self.tr_max_length, self.tr_length * self.tr_grow_factor)
        else:
            self.tr_length = max(self.tr_min_length, self.tr_length * self.tr_shrink_factor)

        self.last_num_evals = n

    def _sample_candidate_pool(self) -> torch.Tensor:
        """
        Sample a candidate pool inside the current trust region using Sobol sequences.
        The result has shape (candidate_pool_size, d) in normalized [0,1] space.
        """
        if self.tr_center is None:
            self.tr_center = torch.full((self.dim,), 0.5, dtype=torch.double)
            self.tr_length = 1.0

        half = self.tr_length / 2.0
        lb = (self.tr_center - half).clamp(0.0, 1.0)
        ub = (self.tr_center + half).clamp(0.0, 1.0)
        width = (ub - lb).clamp(min=1e-6)

        sobol = torch.quasirandom.SobolEngine(self.dim, scramble=True)
        base = sobol.draw(self.candidate_pool_size).to(dtype=torch.double)  # [0,1]^d
        candidates = lb + width * base
        return candidates

    # -------------------------------------------------------------------------
    # Duel-Thompson: winner + challenger
    # -------------------------------------------------------------------------

    def _select_winner_ts(self, candidate_pool: torch.Tensor) -> (torch.Tensor, int):
        """
        Select a winner using posterior-based Thompson sampling.

        Implementation:
        - Draw S samples of f(x) over the candidate pool
        - For each sample, compare f_s[i] against all f_s[j] and count the win ratio of i
        - Average over S samples to estimate each point's mean win rate and choose the maximum as the winner
        """
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(candidate_pool)  # MVN
            f_samples = posterior.rsample(
                sample_shape=torch.Size([self.ts_num_samples])
            ).squeeze(-1)  # (S, N)

            diff = f_samples.unsqueeze(2) - f_samples.unsqueeze(1)  # (S, N, N)
            wins = (diff >= 0).float().mean(dim=2)  # (S, N)
            score = wins.mean(dim=0)  # (N,)

            best_idx = int(torch.argmax(score).item())
            winner_x = candidate_pool[best_idx : best_idx + 1]

        return winner_x, best_idx

    def _select_challenger_by_variance(
        self, winner_x: torch.Tensor, candidate_pool: torch.Tensor
    ) -> torch.Tensor:
        """
        Select the challenger: keep the winner fixed and choose x maximizing the variance of f(winner) - f(x).

        var_diff = var(f_w - f_j) = Σ_ww + Σ_jj - 2 Σ_wj
        """
        self.model.eval()
        with torch.no_grad():
            X_ext = torch.cat([winner_x, candidate_pool], dim=0)  # (1+N, d)
            posterior = self.model.posterior(X_ext)
            cov = posterior.mvn.covariance_matrix  # (1+N, 1+N)

            var_w = cov[0, 0]
            var_all = cov.diag()            # (1+N,)
            var_candidates = var_all[1:]    # (N,)
            cov_w_cand = cov[0, 1:]         # (N,)

            var_diff = var_w + var_candidates - 2.0 * cov_w_cand  # (N,)

            best_idx = int(torch.argmax(var_diff).item())
            challenger = candidate_pool[best_idx : best_idx + 1]

        return challenger

    # -------------------------------------------------------------------------
    # Main interface: keep the original signature suggest_params(self, history)
    # -------------------------------------------------------------------------

    def suggest_params(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Main RankTuner logic (interface unchanged):
        1) Use random sampling for the first self.warmup_trials trials
        2) Filter failed trials and update the trust region
        3) Build pairwise comparisons from history and train PairwiseGP
        4) Sample a candidate pool inside the trust region and run Duel-Thompson:
            - winner: use Thompson sampling to pick the point with the highest average win rate
            - challenger: choose x maximizing the variance of winner - x
        5) Return the parameters corresponding to the challenger
        """
        # 0. Cold start: use random sampling for the first N trials
        if len(history) < self.warmup_trials:
            print("  [RankTuner] Warm-up phase (Random)...")
            return self._random_sample()

        # 1. Filter failed trials (for example, area = inf)
        valid_history = [
            h
            for h in history
            if h.get("metrics", {}).get("area", float("inf")) != float("inf")
        ]
        if len(valid_history) < 2:
            print("  [RankTuner] Not enough valid history, random...")
            return self._random_sample()

        # 2. Update the trust region
        self._update_trust_region(valid_history)

        # 3. Prepare training data for PairwiseGP
        X_train = torch.cat([self._to_tensor(h['params']) for h in history])
        comps = generate_comparisons(history)

        if comps is None or len(comps) == 0:
            print("  [RankTuner] No dominance found yet. Random exploring...")
            return self._random_sample()

        # 4. Train PairwiseGP
        print(f"  [RankTuner] Training PairwiseGP on {len(comps)} comparisons...")
        try:
            self.model = PairwiseGP(X_train, comps, jitter=1e-4)
            mll = PairwiseLaplaceMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"  [Warning] GP Fitting failed: {e}. Fallback to random.")
            return self._random_sample()

        # 5. Duel-Thompson：winner + challenger
        candidate_pool = self._sample_candidate_pool()  # (M, d)
        winner_x, _ = self._select_winner_ts(candidate_pool)
        challenger_x = self._select_challenger_by_variance(winner_x, candidate_pool)

        # The current framework evaluates one point at a time, so return the challenger
        next_x = challenger_x
        return self._to_dict(next_x)

    # -------------------------------------------------------------------------
    # Random sampling (warm-up and fallback)
    # -------------------------------------------------------------------------

    def _random_sample(self) -> Dict:
        params = {}
        for p in self.param_space:
            name = p["name"]
            if "range" in p:
                low, high = p["range"]
                if p.get("type") == "int":
                    val = random.randint(low, high)
                else:
                    val = random.uniform(low, high)
            elif "choices" in p:
                val = random.choice(p["choices"])
            else:
                val = 0
            params[name] = val
        return params
    
    def update(self, *args, **kwargs):
        """
        No-op implementation for compatibility with the TunerAgent abstract interface.

        The current RankTunerAgent logic relies entirely on suggest_params(history)
        and uses the full history passed in, so it does not need extra internal state.
        Therefore, this method intentionally does nothing for now.
        If you later want online updates, incremental logic can be implemented here.
        """
        return


def suggest_params_cost_aware(
    self,
    history: List[Dict],
    runtime_predictor,                 # RuntimePredictor
    *,
    remaining_sec: Optional[float] = None,
    margin_sec: float = 0.0,           # Time reserved for shutdown / finalization
    risk_beta: float = 2.0,            # Conservative runtime estimate using mean + beta * std
    topk_eval: int = 64,               # Only evaluate runtime for the top-k var_diff candidates to reduce overhead
) -> Optional[Dict[str, Any]]:
    """
    Cost-aware version: score = var_diff / (E[time] + beta * Std[time])
    Applies remaining-time gating: (E + beta * Std) < remaining - margin
    """
    # 1) Warm-up: before the GP is available, follow the original logic (or prefer faster options)
    if len(history) < self.warmup_trials:
        return self.suggest_params(history)

    valid_history = [
        h for h in history
        if h.get("metrics", {}).get("area", float("inf")) != float("inf")
    ]
    if len(valid_history) < 2:
        return self.suggest_params(history)

    # 2) Train PairwiseGP (largely reusing the suggest_params logic)
    self._update_trust_region(valid_history)
    X_train = torch.cat([self._to_tensor(h["params"]) for h in history])
    comps = generate_comparisons(history)
    if comps.numel() == 0:
        return self.suggest_params(history)

    self.model = PairwiseGP(X_train, comps)
    mll = PairwiseLaplaceMarginalLogLikelihood(self.model.likelihood, self.model)
    fit_gpytorch_mll(mll, max_retries=3)

    # 3) Sample the candidate pool and select the winner (TS)
    candidate_pool = self._sample_candidate_pool()
    winner_x, _ = self._select_winner_ts(candidate_pool)

    # 4) Compute var_diff for each candidate (same quantity used for the challenger)
    self.model.eval()
    with torch.no_grad():
        X_ext = torch.cat([winner_x, candidate_pool], dim=0)
        posterior = self.model.posterior(X_ext)
        cov = posterior.mvn.covariance_matrix

        var_w = cov[0, 0]
        var_candidates = cov.diag()[1:]
        cov_w_cand = cov[0, 1:]
        var_diff = var_w + var_candidates - 2.0 * cov_w_cand  # (N,)

    # 5) Evaluate runtime only for the top-k var_diff candidates, then select by var_diff / time
    N = var_diff.shape[0]
    k = min(int(topk_eval), N)
    top_idx = torch.topk(var_diff, k=k, largest=True).indices.tolist()

    best_score = -1.0
    best_params = None

    for idx in top_idx:
        cand_x = candidate_pool[idx : idx + 1]
        cand_params = self._to_dict(cand_x)  # Parameter dictionary in the original space

        mean_t, std_t = runtime_predictor.predict(cand_params)
        t_u = mean_t + risk_beta * std_t

        if remaining_sec is not None:
            if t_u > max(0.0, remaining_sec - float(margin_sec)):
                continue

        score = float(var_diff[idx].item()) / max(1.0, float(t_u))
        if score > best_score:
            best_score = score
            best_params = cand_params

    # 6) If no point fits within the remaining time budget, return None so the scheduler stops launching new work
    return best_params
