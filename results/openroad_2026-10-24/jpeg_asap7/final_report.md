# Final DSE Report

## Executive Summary

- **Run inventory:** 135 launched, **135 job_success, 0 job failures, 135 usable**, but only **45 full_valid** (33%). The other 90 are quick/CTS-fidelity points with `metrics_status=partial` (`metrics_valid=false`). Pareto set = **9 full-valid points**.
- **The flow never broke — the search under-delivered.** There are no crash logs (`fail_log_tail` empty everywhere, `job_fail=0`). The real losses are silent QoR regressions and an acquisition signal that saturates early.
- **Hypervolume converged very early.** `cum_hv` reached 1.824e9 at `iter_31` (~1.4 h of wall time) and finished at 1.8424e9 — **~1% of the total HV was won with ~77% of the remaining budget**. Effective convergence needed ~30 of 135 samples.
- **The frontier is real but narrow:** cell area 6396–6649 µm² (3.9% span), power 0.1397–0.1471 W (5.3% span), WNS from **−4.1 to +66.0**. So the achievable trade is roughly *5% power ↔ 40 WNS units*, with area almost a constant.
- **Timing is controlled by two binary knobs, not continuous tuning.** Every full-valid point with WNS < 16 is either `gp_timing_driven=0` (5 points, WNS −8.1…+6.3) or `ABC_AREA=1` combined with a small CTS cluster diameter (2 points). With `gp_timing_driven=1 AND ABC_AREA=0`, **all 33 points have WNS ≥ 20.9 (median ≈ 54)**.
- **Strongest geometric lever:** `CTS_CLUSTER_DIAMETER ≥ 75` with `cts_cluster_size ≤ 26` yields mean WNS ≈ 53; `CTS_CLUSTER_DIAMETER ≤ 60` yields mean WNS ≈ 15 — a ~38-unit separation from one clock-tree knob.
- **Data quality is the weakest link:** the headline `best_wns` (iter_121, 73.10) is a **CTS-stage quick point that was never promoted**, so it is not a validated post-route result. For all 135 runs `clock_period = Infinity`, `ecp = Infinity` (and `wl = Infinity` for all 90 quick runs) — frequency-normalized objectives do not exist. Area and power are **byte-identical between quick and full** (e.g. iter_64/iter_92 = 6648.57 µm²; iter_106/iter_115 = 6431.22 µm²), proving post-route area/power is never actually re-measured: the "power" objective is a placement/CTS estimate only.

## Search Quality and Data Quality

**Search quality (how well the space was explored and ranked)**

| Aspect | Evidence | Assessment |
|---|---|---|
| Flow robustness | 135/135 success, 0 fail stages, no crash tails | Excellent |
| Fidelity discipline | 90 quick (CTS, `4_1_cts.json`) + 45 full (`6_report.json`, `make_target=finish`) | Reasonable 2:1 quick:full ratio |
| Convergence | HV 1.824e9 @iter_31 → 1.8424e9 final; last deltas 172, 6.9k | **Premature saturation**; ~1% HV for 77% of budget |
| Frontier density | 9 Pareto points inside a 3.9% area / 5.3% power box | Locally flat, noise-dominated frontier |
| Coverage | Util sampled 51.5–70.0 (79/135 quick runs in 58–70); AR 0.82–1.20; RLA 0.301–0.60 | **Undersampled:** util < 51 (≈3 pts), AR > 1.20 (0 pts), util > 70 (0 pts) |
| Bound saturation | `CTS_CLUSTER_DIAMETER` = 78–80 in 40+ runs; AR = 1.20 repeatedly; RLA = 0.60 at iter_115 | Optimizer **pinned against box bounds** → wasted samples, low diversity |
| Promotion value | Several promotions are exact duplicates of the quick parent (identical area/power), and 6 promoted runs regressed vs parent by −9.7 to −16.7 WNS | Promotion decision often carried **no new information** |
| Best-point validation | `best_wns` iter_121 (73.1) and `best_area` iter_62 (6396.29) are **both partial/quick** points | Headline bests are **unvalidated** — reporting hazard |

**Data quality (can the numbers be trusted?)**

- **Broken derived fields:** `clock_period = Infinity` and `ecp = Infinity` for all 135 records; `wl = Infinity` for all 90 quick records. Any frequency/ECP-based objective or constraint is silently non-functional. Units are self-described as ambiguous ("ps or ns").
- **No post-route area/power feedback:** area and power come from `4_1_cts.json` for quick and are numerically unchanged in `6_report.json` for full. Consequently area/power Pareto separation reflects *synthesis mapping + placement*, not routing. Power deltas quick→full are tiny (−0.0007 to −0.0018 W), essentially noise.
- **WNS is the only quantity that genuinely changes** from CTS to finish, and it changes a lot: paired quick→full deltas include +19.3 (iter_106→115), −16.7 (iter_116→117), −13.2 (iter_72→90), −11.9 (iter_69→108), −11.0 (iter_67→98), +14.8 (iter_56→96). **Quick-stage WNS is a biased-and-noisy estimator (mean ≈ −2, spread ≈ ±10–17).** Ranking on it directly explains most "regressed vs parent" events.
- **Congestion proxy is degenerate:** `proxy_route_overflow = 430.0` for the overwhelming majority of quick runs (420 for a few, 550 once, and 0.0 for iter_62). Only four runs differentiate (2280, 2180, 1920, 1280). A metric with a constant floor cannot rank candidates; any acquisition relying on it is effectively blind.
- **`symptoms` field is non-discriminating:** 130/135 full_ and quick-stage logs carry `["overflow","congestion"]` — this is a grep artifact of GRT text, not an outcome label. Only two runs carry `["setup"]` (iter_88, iter_62), and those are the only negative-slack results. Net: **there are 0 real failure snippets to diagnose; failure analysis must be done from QoR deltas.**
- **Metric cross-checks that do hold:** power is internally consistent with wirelength within the `ABC_AREA=0` family (WL 221–229k → 0.1467–0.1480 W; WL 322–337k → 0.1561–0.1566 W). So power is *directionally* trustworthy, just not post-route.

## Pareto Frontier Insights

The 9 full-valid Pareto points fall into **three distinct regimes**, not a smooth surface:

**Regime A — Power-minimal via `ABC_AREA=1`** (iter_17, iter_18, iter_92)
- Power 0.13973–0.14024 W (best overall), area 6637.7–6648.6 µm² (**+3.3% vs absolute best**), WNS 12.1–40.6, WL 240.8–244.3k.
- Synthesis area-recovery mapping shrinks/swaps cells → −4.7% power, +3.3% cell area, and a **timing tax of ~20–50 WNS units** relative to Regime B.
- iter_18 is the dominated-looking end of this group (WNS 12.1 for only 0.0005 W less than iter_17): its small `CTS_CLUSTER_DIAMETER=25` and `AR=0.83` cost timing for nothing.

**Regime B — Balanced / WNS-maximal** (iter_31, iter_65, iter_112, iter_115)
- WNS 37.0–66.0 (best), area 6431.2–6436.3 (**+0.6% over best**), power 0.14673–0.14709 (**+5.2% over Regime A**), WL 223.4–228.9k (**lowest**).
- Signature: `CORE_UTILIZATION` 68.5–70.0, `CORE_ASPECT_RATIO` 1.177–1.20, `ROUTING_LAYER_ADJUSTMENT` 0.535–0.60, `cts_cluster_size` 10–16, `CTS_CLUSTER_DIAMETER` 78–80, `gp_timing_driven=1`, `ABC_AREA=0`.
- This is the **efficient part of the frontier**: moving iter_115 → iter_112 buys +14.2 WNS for +1.1 µm² area and +0.00016 W. Essentially free timing.

**Regime C — "Area floor" trap** (iter_88, iter_89)
- Area 6397.2–6404.5 (nominally best) but **worst power 0.1561–0.1566** and **worst wirelength 322.7–337.4k (+45–51%)**, WNS −4.1 and +6.3.
- Drivers: `ROUTING_LAYER_ADJUSTMENT` 0.35/0.3668, `gp_timing_driven=0`, `CORE_UTILIZATION` 56–61, `AR≈0.95–0.99`.
- These are only on the frontier because **wirelength is not an objective**. They are strictly worse than Regime B on every practical axis (power +7%, WL +46%, WNS −30 to −70). Their "area win" is a 0.5% artifact of an area-oriented legality/legacy corner, not a real PPA gain.

**Trade-off summary on the frontier**

| Trade | Magnitude | Direction |
|---|---|---|
| Power ↔ WNS | 5.2% power (0.1397 → 0.1471) buys **+25 to +54 WNS** | Strongly favorable to Regime B |
| Area ↔ WNS | 0.6% area (6431 → 6436) buys **+29 WNS** (iter_65 → iter_112) | Nearly free |
| Area ↔ Power | 3.9% area ↔ 4.7% power (`ABC_AREA` 0→1) | Direct coupling; pick one |
| WL ↔ Power | +51% WL ↔ +6.6% power | Real, and currently unrewarded |

## Parameter Sensitivity

**Dominant (near-deterministic) effects**

1. **`gp_timing_driven` (binary).** All 5 full points with `gp_timing_driven=0` land at WNS ≤ 6.3 (−8.1, −5.9, −4.1, +2.8, +6.3); all 40 with `=1` land at WNS ≥ 12.1. This single knob separates the entire failing population. Non-timing-driven GP also inflates WL to 322–369k.
2. **`ABC_AREA` (binary).** `=1` → area 6637.7–6648.6 and power 0.13973–0.14024 (tight clusters, n=4); `=0` → area 6396–6449 and power 0.1467–0.1569. It is the *only* knob with a clean, monotone area effect (+~210 µm², +3.3%) and the only one that reaches sub-0.141 W.
3. **`CTS_CLUSTER_DIAMETER` (continuous, currently bound-limited).** ≥75 → mean WNS ≈ 53 (n≈19, min 26.4); ≤60 → mean WNS ≈ 15 (n≈9, includes all negative-slack runs). Small diameters (25, 28, 30, 36) are consistently the worst (12.1, 22.5, −8.1, −5.9). Interpretation: tightly bounded clock clustering produces deeper/skew-heavy trees at this 680 ps-class target. **The upper bound (80) is being hit constantly — the optimum is likely outside the sampled box.**
4. **`cts_cluster_size` (interacting).** The winning combination is *small size, large diameter* (10–26 / 78–80). Conversely `cts_cluster_size=40` pairs with WNS 34–49 and WL 236k (iter_96, iter_117).

**Moderate effects**

5. **`CORE_ASPECT_RATIO`.** 1.15–1.20 → WNS typically 50–66; 0.82–0.95 → WNS typically −8 to +34 (mean ≈ 14). But the *smallest areas* (6396.3, 6397.2, 6404.5) come from AR 0.95–0.99: **AR is the one true area-vs-timing dial**, worth ~35 WNS for ~0.6% area. AR = 1.20 is being pinned.
6. **`ROUTING_LAYER_ADJUSTMENT`.** 0.50–0.60 co-occurs with low WL (221–229k) and high WNS; 0.30–0.44 co-occurs with WL 226–369k and WNS −8 to +48. Confidence is limited because RLA is entangled with the CTS regime and with `place_padding`; it is probably acting as a *proxy* for "which regime am I in" rather than an independent lever, but the bound 0.60 is clearly active.
7. **`place_padding`.** `=1` runs show WL 226–244k (and one 337k) and WNS 6.3–62.8; `=0` runs dominate the best WNS/WL pairs (221–229k). Net effect: **+2 to +5% wirelength and a mild WNS penalty** — padding disperses cells without buying routability here.
8. **`gp_routability_driven`.** No measurable benefit: its runs span WNS −8.1 to +62.8 and 4 of the 5 highest `proxy_route_overflow` values (1280–2280) occur with it enabled. Congestion is not the binding constraint; this knob is consuming search budget.

**Weak / no detectable effect**

9. **`PLACE_DENSITY_LB_ADDON`** (0.02–0.25): no monotone trend. Large addon at *low* util is harmful (iter_89: 0.2406 @ util 56.4 → WNS 6.3), consistent with density floor being a function of util, so PDLA should be swept *conditionally*, not independently.
10. **`TNS_END_PERCENT`** (10–100) and **`CORE_MARGIN`** (1–8): no resolvable signal (good points at 14, 17, 21, 95, 100 and bad points at 10, 20, 82). Keep them at defaults/coarse levels; they are wasting dimensions.

**Metric-side sensitivities**
- WNS is the only metric responsive to routing-stage reality; area/power respond to synthesis mapping only.
- Power scales with wirelength inside a fixed `ABC_AREA` family at ≈ +6.6% power per +51% WL — so **WL should be a constraint**, not an afterthought.

## Failure Analysis

**There are no tool failures.** `job_fail = 0`, `fail_stage` empty, `fail_log_tail` empty for all 135 records, and the two `final_status` values are `null` throughout. No `*_error`/`*_fatal` snippets exist to quote. What follows is therefore reconstructed from `symptoms`, `stage.qor_vs_parent`, and metric geometry.

**Failure mode 1 — Silent post-route setup loss ("setup" symptom, 2 runs).**
iter_88 (`WNS −4.08`) and its quick parent iter_62 (`WNS −2.89`) share identical params: `CORE_UTILIZATION=61.0, CORE_ASPECT_RATIO=0.95, CORE_MARGIN=3, PLACE_DENSITY_LB_ADDON=0.05, gp_timing_driven=0, gp_routability_driven=0, place_padding=0, TNS_END_PERCENT=100, cts_cluster_size=20, CTS_CLUSTER_DIAMETER=55, ROUTING_LAYER_ADJUSTMENT=0.35`.
- Critical diagnostic: iter_62's `proxy_route_overflow = 0.0` — **zero congestion with negative slack**. This is unambiguously a *logic/clock-skew* failure, not a routability failure. Root cause = `gp_timing_driven=0` (placement never optimized delay) plus a small CTS diameter (55) and a low RLA (0.35) that removed routing resources from the timing-critical nets.
- Consequence: the point still entered the Pareto set (it owns the area extreme) because the objective set has no wirelength or slack-floor guard.

**Failure mode 2 — Wirelength / power blow-up at the "cheap area" corner (3 runs).**
iter_89 (WL 337,446; P 0.15660), iter_88 (WL 322,704; P 0.15609), iter_19 (WL 336,037; P 0.15687) vs the Regime-B baseline (WL 221–229k; P 0.1467). That is **+45–51% wirelength and +6.6% power for ≤0.5% area**. Signature again: `gp_timing_driven=0` and/or `gp_routability_driven=1` + `place_padding=1` + RLA ≤ 0.42. Note iter_89's `qor_vs_parent` reports WNS **+24.7** — an apparent "improvement" that is pure quick→full noise, since its absolute WNS is 6.3 and its WL exploded.

**Failure mode 3 — Promotion-time ranking inversion (6+ runs).**
Promoted children regressing against their parents: iter_117 (−16.71), iter_63 (−13.95), iter_108 (−11.85), iter_98 (−11.0), iter_104 (−10.73), iter_107 (−9.70), iter_18 (−9.67). In every case the parent's quick-stage WNS was inflated; the paired deltas (−16.7, −13.2, −11.9, −11.0, −10.7, −9.7) are all within the observed quick→full noise band (σ ≈ 10–15 WNS). **These are not design failures; they are measurement-noise failures**, and they consumed 6 of the 45 expensive full runs.

**Failure mode 4 — Metric-extraction defect (all 135 runs).**
`clock_period = Infinity` ⇒ `ecp = clock_period − wns = Infinity`. `wl = Infinity` for all 90 quick runs. Any objective, constraint, filter, or surrogate term built on ECP, frequency, or wirelength was silently inert for the whole campaign. This is the single highest-leverage defect: it made WL-driven power regressions (Mode 2) invisible to the optimizer.

**Failure mode 5 — Degenerate congestion signal (90 quick runs).**
`proxy_route_overflow` is 430.0 for 80+ runs (a floor/clamp), 420.0 for 4, 550.0 once, 0.0 once, and >1000 for only 4 runs (iter_56 = 2280, iter_111 = 2180, iter_123 = 1920, iter_113 = 1280). A near-constant feature contributes zero information, so "congestion symptoms" in the logs cannot be used to reject candidates — and 4 of the 5 genuinely-congested runs (`place_padding=1` + `gp_routability_driven=1`) suggest the congestion that *does* exist is induced by padding, not by utilization.

**Failure mode 6 — Search-space bound stasis.**
`CTS_CLUSTER_DIAMETER` = 78/79/80 in 40+ runs, `CORE_ASPECT_RATIO` = 1.20/1.19 in the top-performing cluster, `ROUTING_LAYER_ADJUSTMENT` = 0.60 at iter_115. The optimizer is repeatedly proposing boundary points, producing 9 Pareto points whose spread (3.9% area, 5.3% power) is smaller than the noise of the one metric (WNS) that actually varies. **Part of the frontier is noise membership, not design merit.**

## Recommended Next Experiments

1. **Promote and validate the top quick-only candidates immediately.** iter_121 (73.10), iter_67 (70.96), iter_105 (70.22), iter_97 (69.29), iter_72 (68.84), iter_124 (65.35). Run these at `make_target=finish`. Given a mean quick→full delta of ≈ −2 with σ ≈ 10–15, expect post-route WNS ≈ 55–65; if iter_121 lands ≥65 it strictly dominates iter_112 and reshapes the frontier. **Do not report 73.10 as a result until validated.**

2. **Fix the metric contract before spending more budget.** Capture `clock_period`/frequency, DRT wirelength, and post-route power for every run; hard-fail (or flag) any record with `Infinity` in a required field, and stop deriving ECP from a missing period. Until this is fixed, area/power Pareto separation is a synthesis artifact.

3. **Re-run Regime B with the bounds opened.** Sweep `CTS_CLUSTER_DIAMETER ∈ [78, 95]`, `cts_cluster_size ∈ [8, 16]`, `CORE_UTILIZATION ∈ [66, 70]`, `CORE_ASPECT_RATIO ∈ [1.15, 1.25]`, `ROUTING_LAYER_ADJUSTMENT ∈ [0.55, 0.65]`, `gp_timing_driven = 1`, `ABC_AREA = 0`, everything else default. This is a ~12–15 point design; expected +5–10 WNS at ≤0.5% area.

4. **Test the untried power×timing combination.** `ABC_AREA=1` has never been paired with the Regime-B geometry (all 4 `ABC_AREA=1` points used util 55–65, AR 0.83–1.11, diameter 25–70). Run 3–4 points at `ABC_AREA=1, util 69–70, AR 1.19, cts_cluster_size 12, CTS_CLUSTER_DIAMETER 80, RLA 0.58`. If the −0.007 W benefit is additive with the +30 WNS geometry gain, this produces the first **0.140 W / WNS ≥ 50** point — a genuine frontier move rather than a reshuffle.

5. **Prune the search space with hard rules.** Forbid `gp_timing_driven = 0` (5/5 full-valid samples gave WNS ≤ 6.3) and forbid `CTS_CLUSTER_DIAMETER < 60` (all negative-slack points are in this corner). Drop `gp_routability_driven` and `TNS_END_PERCENT` from the active dimension set — neither shows a resolvable effect and both consume samples.

6. **Add wirelength as a constraint, not an objective.** Require WL ≤ 240k µm (Regime B/C boundary). This alone removes iter_88 and iter_89 from the Pareto set, eliminates the "cheap area" trap, and prevents the +6.6% power penalty that currently hides inside the power objective.

7. **Kill duplicate and noise-driven promotions.** Deduplicate candidates on `(ABC_AREA, CORE_UTILIZATION, CORE_ASPECT_RATIO, cts_cluster_size, CTS_CLUSTER_DIAMETER, ROUTING_LAYER_ADJUSTMENT)`; several promotions reproduced the parent's area to 0.01 µm². Then require a paired-delta threshold (≥ 1σ ≈ 10 WNS) before recording a regression, and run the top-3 candidates twice (seed replicates) to estimate route noise before final ranking.

8. **Replace the congestion feature or drop it.** `proxy_route_overflow` is a 430.0 floor for ~80
