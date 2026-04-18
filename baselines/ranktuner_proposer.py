#!/usr/bin/env python3
"""
baselines/ranktuner_proposer.py
===============================
RankTuner: Ranking-based Tool Parameter Tuning via Preference Bayesian Optimization.

Reproduces the core algorithm from:
  "RankTuner: When Design Tool Parameter Tuning Meets Preference Bayesian Optimization"
  (Xu et al., CUHK, ICCAD 2024)

Key components:
  1. Pairwise Gaussian Process (§3.2)  — learns Pareto-dominance relationships
  2. Duel-Thompson Sampling  (§3.3)   — acquisition for comparison-based exploration
  3. Random Embedding (§3.4)           — project high-D → low-D for tractable GP
  4. Trust Region                      — focus search around best-known region
  5. GP hyperparameter optimization    — marginal likelihood based lengthscale/outputscale tuning

This module exposes ``RankTunerProposer`` which follows the same interface as
RandomProposer / BOProposer / OptunaProposer in ``run_baseline.py``.

Release notes relative to an earlier internal revision:
  - Added Random Embedding (paper §3.4): projects D-dim param space to d_e-dim
    before fitting GP. Critical for D≥15 — without it GP degenerates.
  - Added GP hyperparameter optimization via grid search on log marginal likelihood.
  - Fixed default lengthscale: now adaptive based on embedding dim, not hardcoded 0.5.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm as sp_norm

# ---------------------------------------------------------------------------
# Random Embedding (Paper §3.4)
# ---------------------------------------------------------------------------

class RandomEmbedding:
    """
    Random linear projection from D-dim normalized space to d_e-dim.

    Paper §3.4: "Random embedding can effectively cut down the number of
    variables in the search process and thus reduce the runtime of BO."

    We use a fixed Gaussian random matrix A ∈ R^{d_e × D}, and project
    x_embed = A @ x_full.  To lift back, we use the pseudo-inverse:
    x_full = A^+ @ x_embed, then clamp to [0,1]^D.

    The embedding is generated once at init and stays fixed for the entire run.
    """

    def __init__(self, D: int, d_e: int, seed: int = 42):
        self.D = D
        self.d_e = d_e
        rng = np.random.RandomState(seed)
        # Gaussian random projection matrix, scaled by 1/sqrt(d_e)
        A = rng.randn(d_e, D).astype(np.float64) / math.sqrt(d_e)
        self.A = torch.tensor(A, dtype=torch.double)          # (d_e, D)
        # Pseudo-inverse for lifting back
        self.A_pinv = torch.linalg.pinv(self.A)                # (D, d_e)

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        """Project X ∈ R^{n×D} → R^{n×d_e}."""
        return X @ self.A.T  # (n, d_e)

    def lift(self, Z: torch.Tensor) -> torch.Tensor:
        """Lift Z ∈ R^{n×d_e} → R^{n×D}, clamped to [0,1]^D."""
        return (Z @ self.A_pinv.T).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Pairwise Gaussian Process  (Paper §3.2)
# ---------------------------------------------------------------------------

class PairwiseGP:
    """
    Pairwise (preference) Gaussian process with Laplace approximation.

    Given comparison data D = {(x_v, x_u, y)} where y=1 means x_v ≻ x_u
    (Pareto-dominates), we learn a latent function f such that

        p(x_v ≻ x_u | f) = Φ( (f(x_v) - f(x_u)) / (√2 σ) )        [Eq.4]

    Posterior is approximated via Laplace:
        p(f | D) ≈ N(f_MAP, (K⁻¹ + Λ_MAP)⁻¹)                        [§3.2]

    We use an RBF (squared-exponential) kernel with learned hyperparameters.
    """

    def __init__(
        self,
        dim: int,
        noise_sigma: float = 1.0,
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        jitter: float = 1e-6,
    ):
        self.dim = dim
        self.noise_sigma = noise_sigma          # σ in Eq.4
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.jitter = jitter

        # Fitted quantities
        self.X_train: Optional[torch.Tensor] = None      # (n, d) unique training points
        self.comparisons: List[Tuple[int, int]] = []      # (winner_idx, loser_idx) into X_train
        self.f_map: Optional[torch.Tensor] = None         # (n,)  MAP latent values
        self.L_post: Optional[torch.Tensor] = None        # (n, n) Cholesky of posterior covariance
        self.K_train: Optional[torch.Tensor] = None       # (n, n) prior kernel matrix

    # --- Kernel --------------------------------------------------------
    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """RBF kernel: k(x, x') = s² exp(-||x-x'||² / (2 l²))"""
        sq_dist = torch.cdist(X1, X2, p=2.0).pow(2)
        return self.outputscale * torch.exp(-0.5 * sq_dist / (self.lengthscale ** 2))

    # --- Log marginal likelihood (for hyperparameter optimization) ------
    def _log_marginal_likelihood(
        self,
        X: torch.Tensor,
        comparisons: List[Tuple[int, int]],
        f: torch.Tensor,
        K: torch.Tensor,
    ) -> float:
        """
        Approximate log marginal likelihood at f_MAP for model selection.
        log p(D|θ) ≈ log p(D|f_MAP,θ) + log p(f_MAP|θ) - 0.5 log|K⁻¹+Λ|

        We use this to compare different hyperparameter settings.
        """
        n = X.shape[0]
        if n == 0 or len(comparisons) == 0:
            return -1e10

        # log p(D|f) — sum of log Φ for each comparison
        log_lik = 0.0
        sqrt2_sigma = math.sqrt(2.0) * self.noise_sigma
        for wi, li in comparisons:
            z = float(f[wi] - f[li]) / sqrt2_sigma
            log_lik += float(sp_norm.logcdf(z))

        # log p(f|θ) — log of Gaussian prior N(0, K)
        try:
            L_K = torch.linalg.cholesky(K + self.jitter * torch.eye(n, dtype=K.dtype))
            alpha = torch.cholesky_solve(f.unsqueeze(1), L_K).squeeze(1)
            log_prior = -0.5 * float(f @ alpha) - float(torch.log(L_K.diagonal()).sum())
        except Exception:
            log_prior = -1e10

        return log_lik + log_prior

    # --- Fit (Laplace approximation) -----------------------------------
    def fit(
        self,
        X_unique: torch.Tensor,
        comparisons: List[Tuple[int, int]],
        max_iter: int = 50,
    ) -> None:
        """
        Fit the pairwise GP via Laplace approximation (Newton steps).

        Args:
            X_unique   : (n, d) tensor of unique training inputs
            comparisons: list of (winner_idx, loser_idx) tuples
            max_iter   : max Newton iterations for MAP
        """
        self.X_train = X_unique.clone()
        self.comparisons = list(comparisons)
        n = X_unique.shape[0]

        if n == 0 or len(comparisons) == 0:
            self.f_map = torch.zeros(n, dtype=torch.double)
            self.K_train = torch.eye(n, dtype=torch.double)
            self.L_post = torch.eye(n, dtype=torch.double)
            return

        K = self._kernel(X_unique, X_unique)
        K += self.jitter * torch.eye(n, dtype=torch.double)
        self.K_train = K

        try:
            L_K = torch.linalg.cholesky(K)
        except Exception:
            K += 1e-4 * torch.eye(n, dtype=torch.double)
            L_K = torch.linalg.cholesky(K)
            self.K_train = K

        # Newton iterations for f_MAP
        f = torch.zeros(n, dtype=torch.double)
        sqrt2_sigma = math.sqrt(2.0) * self.noise_sigma

        for _ in range(max_iter):
            # Compute gradient and Hessian of log p(D|f)
            grad = torch.zeros(n, dtype=torch.double)
            Lambda = torch.zeros(n, n, dtype=torch.double)

            for wi, li in comparisons:
                z = (f[wi] - f[li]) / sqrt2_sigma
                z_val = float(z)

                # Clamp for numerical stability
                pdf_val = max(sp_norm.pdf(z_val), 1e-30)
                cdf_val = max(sp_norm.cdf(z_val), 1e-30)
                ratio = pdf_val / cdf_val          # λ(z) = φ(z)/Φ(z)
                d_ratio = -(ratio * (ratio + z_val))  # λ'(z)

                coeff_g = ratio / sqrt2_sigma
                grad[wi] += coeff_g
                grad[li] -= coeff_g

                coeff_h = d_ratio / (2.0 * self.noise_sigma ** 2)
                Lambda[wi, wi] -= coeff_h
                Lambda[li, li] -= coeff_h
                Lambda[wi, li] += coeff_h
                Lambda[li, wi] += coeff_h

            # Make Lambda PSD
            Lambda = 0.5 * (Lambda + Lambda.T)
            eig_min = float(torch.linalg.eigvalsh(Lambda).min())
            if eig_min < 1e-8:
                Lambda += (1e-8 - eig_min) * torch.eye(n, dtype=torch.double)

            # Newton step: f_new = (K⁻¹ + Λ)⁻¹ (grad + Λ f)
            try:
                W = torch.linalg.solve(K, torch.eye(n, dtype=torch.double)) + Lambda
                W = 0.5 * (W + W.T)
                rhs = grad + Lambda @ f
                f_new = torch.linalg.solve(W, rhs)
            except Exception:
                break

            diff = float((f_new - f).norm())
            f = f_new
            if diff < 1e-6:
                break

        self.f_map = f

        # Posterior covariance: (K⁻¹ + Λ)⁻¹
        try:
            Sigma_inv = torch.linalg.solve(K, torch.eye(n, dtype=torch.double)) + Lambda
            Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)
            Sigma_inv += self.jitter * torch.eye(n, dtype=torch.double)
            self.L_post = torch.linalg.cholesky(torch.linalg.inv(Sigma_inv))
        except Exception:
            self.L_post = 0.1 * torch.eye(n, dtype=torch.double)

    # --- Predict -------------------------------------------------------
    def predict(self, X_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Posterior mean and variance at X_new.

        Returns:
            mu  : (m,) posterior mean
            var : (m,) posterior variance (diagonal)
        """
        if self.X_train is None or self.f_map is None:
            m = X_new.shape[0]
            return torch.zeros(m, dtype=torch.double), torch.ones(m, dtype=torch.double)

        k_star = self._kernel(X_new, self.X_train)  # (m, n)
        K_inv_f = torch.linalg.solve(self.K_train, self.f_map)
        mu = k_star @ K_inv_f

        k_star_star = self._kernel(X_new, X_new).diagonal()
        K_inv_k = torch.linalg.solve(self.K_train, k_star.T)
        var = k_star_star - (k_star * K_inv_k.T).sum(dim=1)
        var = var.clamp(min=1e-10)

        return mu, var

    def predict_preference(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Predict p(x_a ≻ x_b) and its variance.
        """
        X = torch.stack([x_a, x_b]).unsqueeze(0) if x_a.dim() == 1 else torch.stack([x_a.squeeze(), x_b.squeeze()])
        if X.dim() == 3:
            X = X.squeeze(0)
        mu, var = self.predict(X)

        mu_diff = float(mu[0] - mu[1])
        var_diff = float(var[0] + var[1])
        sqrt2_sigma = math.sqrt(2.0) * self.noise_sigma

        z = mu_diff / math.sqrt(var_diff + sqrt2_sigma ** 2 + 1e-10)
        p = float(sp_norm.cdf(z))

        # Variance of the preference probability (via delta method)
        pdf_z = float(sp_norm.pdf(z))
        var_p = (pdf_z ** 2) * var_diff / (var_diff + sqrt2_sigma ** 2 + 1e-10)

        return p, var_p

    def sample_f(self, X: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw n_samples from the posterior f at X.
        Returns: (n_samples, m) tensor.
        """
        mu, var = self.predict(X)
        std = var.sqrt()
        eps = torch.randn(n_samples, X.shape[0], dtype=torch.double)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)


# ---------------------------------------------------------------------------
# GP Hyperparameter Optimization
# ---------------------------------------------------------------------------

def _optimize_gp_hyperparams(
    X: torch.Tensor,
    comparisons: List[Tuple[int, int]],
    dim: int,
    noise_sigma: float = 1.0,
    jitter: float = 1e-6,
) -> Tuple[float, float]:
    """
    Grid search over lengthscale and outputscale to maximize
    approximate log marginal likelihood.

    Returns (best_lengthscale, best_outputscale).
    """
    # Adaptive grid based on embedding dimension
    # For d_e ~ 6, median pairwise distance in [0,1]^d_e ≈ sqrt(d_e/6) ≈ 1.0
    base_ls = math.sqrt(dim / 6.0)
    ls_grid = [base_ls * f for f in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)]
    os_grid = [0.5, 1.0, 2.0, 4.0]

    best_lml = -float("inf")
    best_ls, best_os = ls_grid[3], os_grid[1]  # defaults

    for ls in ls_grid:
        for os_val in os_grid:
            gp = PairwiseGP(dim=dim, noise_sigma=noise_sigma,
                            lengthscale=ls, outputscale=os_val, jitter=jitter)
            try:
                gp.fit(X, comparisons, max_iter=30)
                if gp.f_map is None:
                    continue
                lml = gp._log_marginal_likelihood(X, comparisons, gp.f_map, gp.K_train)
                if lml > best_lml:
                    best_lml = lml
                    best_ls = ls
                    best_os = os_val
            except Exception:
                continue

    return best_ls, best_os


# ---------------------------------------------------------------------------
# Pairwise comparison generation
# ---------------------------------------------------------------------------

def _is_valid_metrics(m: dict) -> bool:
    if not m:
        return False
    try:
        a = float(m.get("area", float("inf")))
        w = float(m.get("wns", -100.0))
        p = float(m.get("power", float("inf")))
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(w) and w > -99.0 and math.isfinite(p)


def _pareto_dominates(m1: Dict[str, Any], m2: Dict[str, Any]) -> bool:
    """
    m1 Pareto-dominates m2 iff m1 is ≤ on all minimise objectives and
    ≥ on all maximise objectives, with strict inequality on at least one.

    Objectives: minimize area, maximize wns, minimize power.
    """
    a1, w1, p1 = float(m1["area"]), float(m1["wns"]), float(m1["power"])
    a2, w2, p2 = float(m2["area"]), float(m2["wns"]), float(m2["power"])
    cond = (a1 <= a2) and (w1 >= w2) and (p1 <= p2)
    strict = (a1 < a2) or (w1 > w2) or (p1 < p2)
    return bool(cond and strict)


def generate_comparisons_from_history(
    history: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    From evaluation history, extract pairwise Pareto-dominance comparisons.

    Returns:
        valid_records : list of records with valid metrics
        comparisons   : list of (winner_idx, loser_idx)
    """
    valid = [
        h for h in history
        if isinstance(h, dict)
        and isinstance(h.get("params"), dict)
        and h.get("params")
        and _is_valid_metrics(h.get("metrics", {}))
    ]

    comparisons: List[Tuple[int, int]] = []

    # Cap comparisons for very long histories to avoid quadratic growth.
    max_n = len(valid)
    if max_n > 200:
        # Use only the most recent 200 records to avoid O(n²) blowup
        valid = valid[-200:]
        max_n = len(valid)

    for i in range(max_n):
        for j in range(i + 1, max_n):
            mi = valid[i].get("metrics", {})
            mj = valid[j].get("metrics", {})
            if _pareto_dominates(mi, mj):
                comparisons.append((i, j))   # i wins over j
            elif _pareto_dominates(mj, mi):
                comparisons.append((j, i))   # j wins over i
            # else: non-dominated pair — no comparison generated

    return valid, comparisons


# ---------------------------------------------------------------------------
# RankTuner Proposer
# ---------------------------------------------------------------------------

class RankTunerProposer:
    """
    RankTuner: Ranking-based EDA parameter tuning via Preference BO.

    Implements the full pipeline from the paper:
      1. Build pairwise comparisons from Pareto-dominance history
      2. Random Embedding: project D-dim → d_e-dim  (§3.4)
      3. Fit Pairwise GP in embedded space with optimized hyperparameters
      4. Duel-Thompson Sampling acquisition (§3.3):
         - Step 1: sample f̃, pick x_next via soft-winner scoring (Eq.12)
         - Step 2: pick x'_next via maximum V[σ(f*)] (Eq.13)
      5. Trust Region to focus search (in embedded space)

    Interface matches RandomProposer / BOProposer / OptunaProposer.
    """

    def __init__(
        self,
        param_space: List[Dict[str, Any]],
        *,
        noise_sigma: float = 1.0,
        lengthscale: float = 1.0,       # Auto-tuned in practice; default kept at 1.0.
        outputscale: float = 1.0,
        n_candidate_pool: int = 512,
        n_thompson_samples: int = 1,
        min_data_for_gp: int = 6,
        # Random Embedding
        embed_dim: Optional[int] = None,  # ★ NEW: d_e, default = min(D, max(4, D//3))
        embed_seed: int = 42,
        # Trust region
        tr_init_length: float = 0.8,
        tr_min_length: float = 0.05,
        tr_max_length: float = 1.0,
        tr_shrink: float = 0.7,
        tr_grow: float = 1.3,
        # GP hyperparameter optimization
        optimize_hyperparams: bool = True,  # ★ NEW
        hyperparam_opt_interval: int = 5,   # re-optimize every N calls
    ):
        self.param_space = param_space
        self.param_names = [p["name"] for p in param_space]
        self.dim = len(param_space)
        self.bounds = self._get_bounds()

        self.noise_sigma = noise_sigma
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.n_candidate_pool = n_candidate_pool
        self.n_thompson_samples = n_thompson_samples
        self.min_data_for_gp = min_data_for_gp

        # Random Embedding (§3.4)
        if embed_dim is None:
            # Heuristic: d_e ∈ [4, 8] for typical D=15-25
            self.embed_dim = min(self.dim, max(4, self.dim // 3))
        else:
            self.embed_dim = min(embed_dim, self.dim)

        if self.embed_dim < self.dim:
            self.embedding = RandomEmbedding(self.dim, self.embed_dim, seed=embed_seed)
            print(f"  [RankTuner] Random embedding: {self.dim}D -> {self.embed_dim}D")
        else:
            self.embedding = None  # No embedding needed if d_e >= D
            self.embed_dim = self.dim

        # GP hyperparameter optimization
        self.optimize_hyperparams = optimize_hyperparams
        self.hyperparam_opt_interval = hyperparam_opt_interval
        self._propose_count = 0
        self._cached_ls: Optional[float] = None
        self._cached_os: Optional[float] = None

        # Trust Region (operates in embedded space)
        self.tr_center: Optional[torch.Tensor] = None  # (embed_dim,) in embedded space
        self.tr_length = tr_init_length
        self.tr_min_length = tr_min_length
        self.tr_max_length = tr_max_length
        self.tr_shrink = tr_shrink
        self.tr_grow = tr_grow
        self.prev_best_f: Optional[float] = None

        # GP model (re-fit each call)
        self.gp: Optional[PairwiseGP] = None

        # Fallback random
        self._rng = random.Random()

    # --- Bounds / encode / decode ----------------------------------------
    def _get_bounds(self) -> torch.Tensor:
        lower, upper = [], []
        for p in self.param_space:
            if "range" in p:
                lower.append(float(p["range"][0]))
                upper.append(float(p["range"][1]))
            elif "choices" in p:
                lower.append(0.0)
                upper.append(float(max(0, len(p["choices"]) - 1)))
            else:
                lower.append(0.0)
                upper.append(1.0)
        return torch.tensor([lower, upper], dtype=torch.double)

    def _to_tensor(self, params: Dict[str, Any]) -> torch.Tensor:
        """Encode parameter dict → normalized (1, D) tensor in [0,1]^D."""
        raw_vals = []
        for i, name in enumerate(self.param_names):
            p_cfg = self.param_space[i]
            v = params.get(name, 0.0)
            if "choices" in p_cfg:
                choices = list(p_cfg["choices"])
                try:
                    v = float(choices.index(v))
                except (ValueError, TypeError):
                    v = 0.0
            raw_vals.append(float(v))
        t = torch.tensor([raw_vals], dtype=torch.double)
        lo = self.bounds[0]
        hi = self.bounds[1]
        width = (hi - lo).clamp(min=1e-9)
        return ((t - lo) / width).clamp(0.0, 1.0)

    def _to_dict(self, x_norm: torch.Tensor) -> Dict[str, Any]:
        """Decode normalized (1, D) tensor → parameter dict."""
        if x_norm.dim() == 1:
            x_norm = x_norm.unsqueeze(0)
        lo = self.bounds[0]
        hi = self.bounds[1]
        raw = lo + x_norm * (hi - lo)
        raw = raw.squeeze(0)

        params: Dict[str, Any] = {}
        for i, name in enumerate(self.param_names):
            p_cfg = self.param_space[i]
            val = float(raw[i])
            if "choices" in p_cfg:
                choices = list(p_cfg["choices"])
                idx = int(round(max(0.0, min(val, len(choices) - 1))))
                params[name] = choices[idx]
            elif p_cfg.get("type") in ("int", "integer"):
                params[name] = int(round(val))
            else:
                params[name] = round(val, 4)
        return params

    def _random_sample(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for p in self.param_space:
            name = p["name"]
            if "range" in p:
                lo, hi = p["range"]
                if p.get("type") in ("int", "integer"):
                    params[name] = self._rng.randint(int(lo), int(hi))
                else:
                    params[name] = round(self._rng.uniform(float(lo), float(hi)), 4)
            elif "choices" in p:
                params[name] = self._rng.choice(p["choices"])
            else:
                params[name] = 0.0
        return params

    # --- Embedding helpers -----------------------------------------------
    def _embed(self, X_full: torch.Tensor) -> torch.Tensor:
        """Project from full D-dim to embedded d_e-dim."""
        if self.embedding is not None:
            return self.embedding.embed(X_full)
        return X_full

    def _lift(self, Z_embed: torch.Tensor) -> torch.Tensor:
        """Lift from embedded d_e-dim back to full D-dim."""
        if self.embedding is not None:
            return self.embedding.lift(Z_embed)
        return Z_embed.clamp(0.0, 1.0)

    # --- Trust Region (in embedded space) --------------------------------
    def _sample_trust_region(self, n: int) -> torch.Tensor:
        """Sample n points from TR in embedded space, return (n, d_e)."""
        d = self.embed_dim
        if self.tr_center is None:
            self.tr_center = torch.zeros(d, dtype=torch.double)

        half = 0.5 * self.tr_length
        lo = (self.tr_center - half)
        hi = (self.tr_center + half)

        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        u = engine.draw(n).to(dtype=torch.double)
        return lo + u * (hi - lo)

    def _update_trust_region(self, X_embed: torch.Tensor, f_vals: torch.Tensor) -> None:
        """Update TR center and length based on best latent value (in embed space)."""
        if X_embed.shape[0] == 0:
            return
        best_idx = int(torch.argmax(f_vals).item())
        self.tr_center = X_embed[best_idx].clone()

        best_f = float(f_vals[best_idx])
        if self.prev_best_f is not None:
            if best_f > self.prev_best_f + 1e-4:
                self.tr_length = min(self.tr_length * self.tr_grow, self.tr_max_length)
            else:
                self.tr_length = max(self.tr_length * self.tr_shrink, self.tr_min_length)
        self.prev_best_f = best_f

    # --- Duel-Thompson Sampling (§3.3) -----------------------------------
    def _soft_winner_score(
        self,
        f_sample: torch.Tensor,
        X_pool: torch.Tensor,
        n_opponents: int = 64,
    ) -> torch.Tensor:
        """
        Compute soft-winner score (Eq.11/12):
        EI(x) ≈ fraction of random opponents x' where f̃(x) > f̃(x').
        """
        n = X_pool.shape[0]
        opp_indices = torch.randint(0, n, (n_opponents,))
        opp_vals = f_sample[opp_indices]  # (n_opponents,)
        # Vectorized: scores[i] = mean(f_sample[i] > opp_vals)
        scores = (f_sample.unsqueeze(1) > opp_vals.unsqueeze(0)).float().mean(dim=1)
        return scores

    def _exploration_variance(
        self,
        gp: PairwiseGP,
        x_next: torch.Tensor,
        X_pool: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute V[σ(f*)] for each candidate x' paired with x_next.  [Eq.10/13]
        """
        n = X_pool.shape[0]
        var_scores = torch.zeros(n, dtype=torch.double)
        for i in range(n):
            _, var_s = gp.predict_preference(x_next, X_pool[i])
            var_scores[i] = var_s
        return var_scores

    # --- Main propose interface ------------------------------------------
    def propose(self, history: List[Dict], n: int = 1) -> List[Dict[str, Any]]:
        """
        Propose n candidate parameter configurations using RankTuner.
        Falls back to random sampling when insufficient data.
        """
        self._propose_count += 1

        # Extract valid records and pairwise comparisons
        valid, comparisons = generate_comparisons_from_history(history)

        # Not enough data → random
        if len(valid) < self.min_data_for_gp or len(comparisons) < 3:
            return [self._random_sample() for _ in range(n)]

        # Build X tensor (full D-dim, normalized)
        X_full = torch.cat(
            [self._to_tensor(v["params"]) for v in valid], dim=0
        ).to(torch.double)

        # Project to the embedded space.
        X_embed = self._embed(X_full)

        # Periodically optimize GP hyperparameters.
        use_ls = self.lengthscale
        use_os = self.outputscale
        if self.optimize_hyperparams:
            if (self._cached_ls is None
                    or self._propose_count % self.hyperparam_opt_interval == 1):
                try:
                    use_ls, use_os = _optimize_gp_hyperparams(
                        X_embed, comparisons, dim=self.embed_dim,
                        noise_sigma=self.noise_sigma,
                    )
                    self._cached_ls = use_ls
                    self._cached_os = use_os
                    print(f"  [RankTuner] HP optimized: ls={use_ls:.3f}, os={use_os:.3f}")
                except Exception as e:
                    print(f"  [RankTuner] HP opt failed: {e}, using defaults")
                    use_ls = self._cached_ls or self.lengthscale
                    use_os = self._cached_os or self.outputscale
            else:
                use_ls = self._cached_ls or self.lengthscale
                use_os = self._cached_os or self.outputscale

        # Fit Pairwise GP in embedded space
        try:
            gp = PairwiseGP(
                dim=self.embed_dim,
                noise_sigma=self.noise_sigma,
                lengthscale=use_ls,
                outputscale=use_os,
            )
            gp.fit(X_embed, comparisons, max_iter=50)
            self.gp = gp
        except Exception as e:
            print(f"  [RankTuner] GP fit failed: {e}, fallback random")
            return [self._random_sample() for _ in range(n)]

        # Update trust region in embedded space
        if gp.f_map is not None and gp.f_map.numel() > 0:
            self._update_trust_region(X_embed, gp.f_map)

        candidates = []
        for _ in range(n):
            try:
                cand = self._duel_thompson_sample(gp, X_embed)
                candidates.append(cand)
            except Exception as e:
                print(f"  [RankTuner] Thompson sampling failed: {e}, fallback random")
                candidates.append(self._random_sample())

        return candidates

    def _duel_thompson_sample(
        self,
        gp: PairwiseGP,
        X_train_embed: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Duel-Thompson Sampling (§3.3):

        Step 1 (Eq.12): Sample f̃ from posterior, pick x_next = argmax soft_winner(x)
        Step 2 (Eq.13): Pick x'_next = argmax V[σ(f*)] near x_next
        Return the better of x_next, x'_next by latent f_MAP (with exploration stochasticity).

        All operations in embedded space; lift back to full space at the end.
        """
        # Sample candidate pool from trust region (in embedded space)
        Z_pool = self._sample_trust_region(self.n_candidate_pool)

        # --- Step 1: Thompson sample → soft-winner score ---
        f_samples = gp.sample_f(Z_pool, n_samples=1).squeeze(0)
        scores = self._soft_winner_score(f_samples, Z_pool, n_opponents=64)
        x_next_idx = int(torch.argmax(scores).item())
        z_next = Z_pool[x_next_idx]

        # --- Step 2: Exploration — find z' with max V[σ(f*)] (Eq.13) ---
        n_explore = min(128, Z_pool.shape[0])
        explore_indices = torch.randperm(Z_pool.shape[0])[:n_explore]
        Z_explore = Z_pool[explore_indices]

        var_scores = self._exploration_variance(gp, z_next, Z_explore)
        z_prime_idx = int(torch.argmax(var_scores).item())
        z_prime = Z_explore[z_prime_idx]

        # Pick winner by predicted latent value, with exploration stochasticity
        mu_pair, _ = gp.predict(torch.stack([z_next, z_prime]))
        if self._rng.random() < 0.3:
            # 30% chance: return explorer (challenger) for diversity
            chosen_z = z_prime
        elif float(mu_pair[0]) >= float(mu_pair[1]):
            chosen_z = z_next
        else:
            chosen_z = z_prime

        # Lift from the embedded space back to the full space and decode.
        chosen_full = self._lift(chosen_z.unsqueeze(0))  # (1, D)
        return self._to_dict(chosen_full)

    # --- State serialization for resume/checkpoint -----------------------
    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "tr_length": float(self.tr_length),
            "prev_best_f": float(self.prev_best_f) if self.prev_best_f is not None else None,
            "cached_ls": self._cached_ls,
            "cached_os": self._cached_os,
            "propose_count": self._propose_count,
        }
        if self.tr_center is not None:
            state["tr_center"] = self.tr_center.tolist()
        else:
            state["tr_center"] = None
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        if not state or not isinstance(state, dict):
            return

        tr_length = state.get("tr_length")
        if tr_length is not None:
            self.tr_length = float(tr_length)

        prev_best = state.get("prev_best_f")
        if prev_best is not None:
            self.prev_best_f = float(prev_best)
        else:
            self.prev_best_f = None

        tr_center = state.get("tr_center")
        if tr_center is not None and isinstance(tr_center, (list, tuple)):
            self.tr_center = torch.tensor(tr_center, dtype=torch.double)
        else:
            self.tr_center = None

        # Restore cached hyperparameters.
        self._cached_ls = state.get("cached_ls")
        self._cached_os = state.get("cached_os")
        self._propose_count = int(state.get("propose_count", 0))

    @property
    def strategy_name(self) -> str:
        return "ranktuner"


# ---------------------------------------------------------------------------
# Resume helper (called by _restore_proposer_on_resume in run_baseline.py)
# ---------------------------------------------------------------------------

def _restore_ranktuner_proposer(
    proposer: RankTunerProposer,
    history: List[Dict[str, Any]],
    saved_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Restore RankTunerProposer state after a resume.

    Two sources of restoration:
      1. saved_state (from resume_state.baseline.json → "proposer_state")
         — restores Trust Region center/length/prev_best_f exactly.
      2. history (from resume_state → "history_all")
         — re-derive TR state by re-fitting GP on history.

    Source 1 is preferred when available; source 2 is the fallback.
    """
    # (A) Direct state restore from checkpoint
    if saved_state and isinstance(saved_state, dict):
        proposer.set_state(saved_state)
        print(f"  [Resume] RankTuner: restored TR state (length={proposer.tr_length:.4f})")
        return True

    # (B) Fallback: rebuild TR from history by doing a "dry" GP fit
    if not history:
        print(f"  [Resume] RankTuner: no history, starting fresh")
        return False

    valid, comparisons = generate_comparisons_from_history(history)
    if len(valid) < proposer.min_data_for_gp or len(comparisons) < 3:
        print(f"  [Resume] RankTuner: insufficient data for TR rebuild "
              f"(valid={len(valid)}, comps={len(comparisons)}), starting fresh")
        return False

    try:
        X_full = torch.cat(
            [proposer._to_tensor(v["params"]) for v in valid], dim=0
        ).to(torch.double)

        X_embed = proposer._embed(X_full)

        gp = PairwiseGP(
            dim=proposer.embed_dim,
            noise_sigma=proposer.noise_sigma,
            lengthscale=proposer._cached_ls or proposer.lengthscale,
            outputscale=proposer._cached_os or proposer.outputscale,
        )
        gp.fit(X_embed, comparisons, max_iter=50)

        if gp.f_map is not None and gp.f_map.numel() > 0:
            proposer._update_trust_region(X_embed, gp.f_map)
            print(f"  [Resume] RankTuner: rebuilt TR from history "
                  f"(length={proposer.tr_length:.4f}, center set)")
            return True
    except Exception as e:
        print(f"  [Resume] RankTuner: TR rebuild failed: {e}")

    print(f"  [Resume] RankTuner: using default TR (no state restored)")
    return False
