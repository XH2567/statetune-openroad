# Final DSE Report

## Executive Summary

The campaign (`openroad_jpeg_ng45`, nangate45 / `jpeg_encoder`) completed 147 trials with 147 harness-level successes, 145 usable metric sets, 40 full-flow-valid points and 10 Pareto points. **Timing was never closed**: every full-flow point has negative WNS, ranging from −0.0867 ns (iter_60) to −0.339 ns (iter_102) against a 1.0 ns clock, i.e. the PD knobs alone bottom out at ~8–9 % of the clock period. The reachable design space is narrow in area (83.4–91.3 kµm², ~9 % spread) but wide in timing (5× spread in WNS), and **power is almost collinear with area** (0.474–0.618 W), so the true frontier is essentially a WNS-vs-area trade with power as a passenger.

The dominant discovery is a **categorical-synthesis effect, not a placement effect**: `ABC_AREA=1` buys ~0.09 ns WNS for ~+6 % area and ~+14 % power, and it only pays off when combined with `gp_timing_driven=1`. Every `gp_timing_driven=0` point is worse than −0.27 ns. Search quality has plateaued: cumulative hypervolume moved from 219.66 M to 220.23 M over the whole run (~0.26 %), the last ~90 trials contributed <0.03 %, and all 40 full-fidelity promotions were re-evaluations of already-sampled quick configurations.

## Search Quality and Data Quality

**Search quality (real optimization progress)**

| Signal | Observation | Interpretation |
|---|---|---|
| Hypervolume | 219.66 M → 220.23 M (iter_27 → iter_145) | ~0.26 % total gain; frontier essentially found in the first ~1/3 of the run |
| EHVI score | 165.6 (iter_27) → 1–12 (later points) | Marginal value of new points collapsed by ~2 orders of magnitude |
| Pareto size | 4 → 14 → 10 (final) | Frontier churned but did not expand; later points are near-duplicates |
| Promotion efficiency | 96 quick → 40 full; 10/40 full points Pareto (25 %) | Non-trivial hit rate, but 40 full runs produced only ~10 distinct frontier entries |
| Exploration coverage | util 45.5–70, aspect 0.82–1.20, pad 0.02–0.25, RLA 0.30–0.60 | Search stayed inside a narrow box; no low-util (<45) or high-util (>70) region explored |
| Redundancy | iter_102 ≡ iter_18, iter_106 ≡ iter_21, iter_137 ≡ iter_90, iter_135 ≡ iter_125, iter_145 ≡ iter_126 | Full-fidelity stage mostly *verified* quick winners rather than exploring; **no replicate runs**, so noise is unquantified |

**Data quality (metric trustworthiness)**

- `metrics_status`: 40 full, 105 partial (quick/CTS), 2 invalid. All 105 quick rows have `wl = Infinity` (wire length only parsed from the final flow), so any WL-based analysis is full-run-only.
- `clock_period` is `Infinity` in **every** record, so the derived `ecp = clock_period − wns` is meaningless everywhere. Treat ECP as unavailable.
- `best_wns` (−0.0822, iter_38), `best_area` (83 441, iter_18) and `best_power` (0.4743 W, iter_21) are sourced from `4_1_cts.json` — **post-CTS with wire-load models** (see `EST-0027 no estimated parasitics, using wire load models`). They are not signoff-comparable to the full-flow `6_report.json` numbers. The true full-flow best WNS is iter_60 at −0.0867 ns.
- Stage bias is systematic: promoted full runs show area **+130 … +740 µm²** vs their quick parents (e.g. iter_90 85 602 → iter_137 85 857; iter_18 83 441 → iter_102 84 181) and WNS shifts of ±0.01 ns (occasionally +0.057 ns). This is why `qor_vs_parent.qor_regressed = true` for **40/40** full promotions — an artifact of comparing CTS-stage optimistics against final metrics, not a real failure of every trial.
- Quick-stage WNS is a usable but noisy proxy: |ΔWNS| ≤ ~0.013 ns for well-behaved pairs (iter_125→iter_135: −0.1775 → −0.1902), so quick ranking should be treated as ~±0.015 ns accurate, not exact.

Bottom line: the data is internally consistent and highly usable for parameter attribution, but **stage-mixed headline metrics and the broken ECP/clock parsing must be fixed before trusting "best" rankings**.

## Pareto Frontier Insights

The 10 full-valid Pareto points separate into three clusters with a clear cost structure:

| Cluster | Members | WNS (ns) | Area (µm²) | Power (W) | WL (µm) | Price of admission |
|---|---|---|---|---|---|---|
| **Timing-optimized** | iter_60, iter_71, iter_145 | −0.087 … −0.095 | 90 700–90 840 | 0.553–0.557 | 863 k–936 k | **+6.3 % area, +13.8 % power, +16 % WL** vs. mid cluster |
| **Balanced** | iter_137, iter_88, iter_135, iter_27, iter_94 | −0.169 … −0.194 | 85 436–85 901 | 0.4875–0.4909 | 728 k–787 k | Reference point |
| **Area/power-optimized** | iter_106, iter_102 | −0.275 … −0.339 | 84 181–84 212 | 0.480–0.488 | 1 049 k–1 080 k | −1.5 % area for **+0.09 … +0.15 ns WNS loss** and +40 % WL |

Key trade-off facts:

1. **WNS is expensive, area is cheap.** Moving from the mid cluster to the timing cluster costs ~5 200 µm² (+6 %) and ~0.067 W (+13.7 %) for 0.08 ns (~8 % of period). Moving the other way saves only ~1 300 µm² (−1.5 %) but loses 0.09–0.15 ns. The frontier is asymmetric: **the area end is economically unattractive** — trading 1.5 % area for a 15 % slack regression is a poor deal unless area is hard-capped.
2. **The area-optimal corner is saturated by a bad mode**, not by good tuning: 83.4 kµm² (iter_18) and 83.8 kµm² (iter_21) were only reachable with `gp_timing_driven=0`, which is exactly what destroys timing. There is no Pareto point below 84.1 kµm² that retains sub-0.2 ns WNS — a genuine gap worth attacking.
3. **Power is not an independent axis.** Power ≈ 0.474 + 1.26 × 10⁻⁵ · (Area) W across the clean points; the only outliers (iter_20: 0.612 W @ 89.8 kµm²; iter_23: 0.619 W @ 89.8 kµm²) both carry 1.31 M / 1.41 M µm of wire — i.e., **wire length, not cell count, is the power penalty term**. Timing-cluster points pay their power premium partly through 860–936 kµm of wire.
4. **Wire length is a congestion/route-quality proxy, not the WNS driver.** Short-WL points (727–787 k) span the entire mid cluster (−0.17 to −0.19), while the fastest points have 863–936 k. Suppressing WL alone does not buy slack.
5. **The frontier has no knee near closure.** Best WNS is −87 ps on a 1 ns clock with `max_fanout 10` buffering in a 64-MAC + divider design. The PD-tunable floor (~−0.08 ns) strongly suggests the residual violation is structural (DCT MAC adder tree / iterative divider / `jpeg_qnr`→RLE control), not placement-fixable.

## Parameter Sensitivity

**Strong, high-confidence effects**

- **`ABC_AREA` (binary, dominant).** `1` → area ≈ 90.7–91.1 kµm², WNS ≈ −0.087…−0.113, power ≈ 0.553–0.562 W (iter_60, 71, 87, 115, 120, 143, 145, plus 17, 19). `0` → area ≈ 84.2–86.2 kµm², WNS ≈ −0.17…−0.40, power ≈ 0.480–0.495 W. A clean ~+6 % area / ~+0.09 ns step — the single biggest lever in the dataset. Two exceptions (iter_20, iter_23: `ABC_AREA=1` but −0.275/−0.364 ns and 0.61 W) are both congestion-crippled (WL 1.31–1.41 M), i.e. the effect can be erased by bad placement density.
- **`gp_timing_driven` (near-disqualifying when off).** All 8 full-valid points with `gp_timing_driven=0` have WNS ≤ −0.275 ns (iter_12 −0.393, iter_16 −0.362, iter_18 −0.396, iter_20 −0.275, iter_21 −0.278, iter_23 −0.364, iter_102 −0.339, iter_106 −0.275). Every point better than −0.20 ns is timing-driven. The area "saving" from disabling it is only 1.5–6 %, so **it is strictly dominated** in any timing-relevant objective.
- **`ABC_AREA` × `gp_timing_driven` interaction.** Best timing requires both on; the area corner is only reachable with both off (or timing off alone). The two knobs are the reason the frontier splits into three discrete clusters instead of a smooth curve.

**Medium-confidence effects**

- **CTS cluster geometry (`cts_cluster_size`, `CTS_CLUSTER_DIAMETER`).** Small clusters (10–20) track the shortest wire: iter_135 (10/50) 727.6 k, iter_99 (10/50) 739.1 k, iter_35 (14/74) 743.0 k, iter_137 (14/141) 749.7 k, iter_127 (10/50) 750.1 k. Large clusters (33–40) or large diameters with mid clusters drift up: iter_102 (36/97) 1.080 M, iter_12 (38/117) 1.099 M, iter_45 (22/136) 1.010 M. Effect on WL is ~±20–30 %, on power ~±0.01 W at fixed area.
- **`ROUTING_LAYER_ADJUSTMENT`.** Low values combine with long wire and high overflow: iter_102 (0.385 → 1.080 M), iter_18/iter_102 both ~1.08 M; the 727–787 k group sits at 0.48–0.565. Recommend ≥0.45 as a floor; >0.55 shows no further WL benefit (iter_60 0.594 → 863 k, still long due to ABC=1 buffering).
- **`CORE_UTILIZATION` / placement density.** Area is nearly flat over util 55–69 in the `ABC_AREA=0, td=1` family (85.4–86.2 k), so utilization mainly buys congestion risk: util ≥ 68 with small margin and low pad produced overflow proxies of 560–570 and the two hard failures. The feasible interior is ~55–67.
- **`PLACE_DENSITY_LB_ADDON`.** Safe band observed 0.055–0.25, but it is *not* independently safe: the two `FLW-0024` aborts occurred at pad = 0.04 (util 68.8) and pad = 0.25 (util 70.0). It must be checked against computed density, not tuned in isolation.
- **`CORE_ASPECT_RATIO`.** Weak but suggestive: 0.82–0.97 appears in the short-WL group (iter_135 0.82 → 727 k; iter_88 0.875 → 758 k; iter_127 1.20 → 750 k contradicts), while 1.16–1.20 appears in the long-WL group (iter_143 → 971 k, iter_115 → 955 k, iter_71 → 936 k). Treat as a secondary lever at best, confounded with `ABC_AREA`.

**No detectable effect in this dataset:** `CORE_MARGIN` (1–8), `place_padding` (0/1), `TNS_END_PERCENT` (10–100 — no monotonic WNS/area trend), and `gp_routability_driven` (helps overflow, hurts WNS whenever timing-driven is off: it appears in 6 of the 8 worst-timing points).

## Failure Analysis

All 147 trials returned harness success (`job_fail = 0`), but this **overstates flow success**: 2 trials aborted the flow at `3_1_place_gp_skip_io` with `make ... rc=2`, and 105 trials are "partial" (CTS-stage only, WL missing). No DRC, no router crash, no ABC hang, no missing liberty — the failure surface is very narrow.

**Failure mode 1 — illegal computed placement density (`FLW-0024`), 2/147 (1.4 %).**
Snippets:
- iter_52: `[ERROR FLW-0024] Place density exceeds 1.0 (current PLACE_DENSITY_LB_ADDON = 0.040000)` at util 68.8, margin 2, `ABC_AREA=1`.
- iter_107: `[ERROR FLW-0024] Place density exceeds 1.0 (current PLACE_DENSITY_LB_ADDON = 0.250000)` at util 70.0, margin 1, `ABC_AREA=1`.

Both are high-utilization / low-margin configurations where the derived global-placement density crosses 1.0 regardless of the addon value. The message blaming `PLACE_DENSITY_LB_ADDON` is misleading — the guard is on the *computed* density. Fix: precompute `density = f(CORE_UTILIZATION, CORE_MARGIN, addon)` and clamp to ≤0.94 before launch; reject util > 69 with margin < 3 outright.

**Failure mode 2 — pre-existing setup violations that resizing cannot close (`RSZ-0062`), 147/147.**
`[WARNING RSZ-0062] Unable to repair all setup violations.` appears in every log tail examined, and every run's `symptoms` includes `setup`. This is the design's true bottleneck (DCT MAC / divider), not a tool failure, and it is what makes the WNS axis saturate at −0.08 ns.

**Failure mode 3 — optimistic timing from wire-load models (`EST-0027`), all quick runs.**
`[WARNING EST-0027] no estimated parasitics. Using wire load models.` — combined with sourcing metrics from `4_1_cts.json`, this explains why quick-stage WNS/area consistently disagree with full-flow results and why 40/40 promotions flag as "regressed". Not a crash, but a systematic measurement bias larger than most of the parameter effects being ranked on.

**Benign/noise:** `IFP-0028` core-area snap, `PDN-1051 No instances found for grid (CORE_macro_grid_*)` (no macros in this design), and the Verilog `Ignoring module ... contains processes` warnings — none correlate with bad outcomes.

## Recommended Next Experiments

1. **Repair the metric pipeline before further search.** Parse `clock_period`/clock uncertainty so `ecp` is real, store wire length for quick runs, and tag every metric with its producing stage (`4_1_cts.json` vs `6_report.json`). Recompute `qor_vs_parent` only within the same stage; otherwise the 40/40 "regressed" flag will keep mis-signalling the sampler.
2. **Hard-ban `gp_timing_driven=0`** for any point that will be compared on timing. It is strictly dominated: it buys 1.5–6 % area and costs 0.09–0.30 ns. Reserve it only for a pure-area reference curve, clearly labelled.
3. **Fill the frontier gap between 84.2 kµm² and 85.4 kµm².** That band is currently occupied only by poor-timing points. Launch a full-fidelity DOE at `ABC_AREA=0, gp_timing_driven=1`, util 62–66, `PLACE_DENSITY_LB_ADDON` 0.15–0.21, RLA 0.48–0.55, aspect 0.9–1.0, cts 16–24 / 60–90. Target: ≥85.0 kµm² at WNS ≤ −0.16 ns, which would dominate iter_88/iter_27.
4. **Break the binary `ABC_AREA` step with area recovery.** Keep `ABC_AREA=1` for synthesis quality but add an explicit area-recovery pass (`repair_design -max_utilization`, slew/cap margins, `set_wire_rc` before resizing) and re-measure. The current 6 % area / 0.09 ns cliff is almost certainly compressible into intermediate frontier points — the biggest available HV win.
5. **Lock wire length down as a controlled variable.** Adopt `cts_cluster_size ≤ 20`, `CTS_CLUSTER_DIAMETER 50–80`, `ROUTING_LAYER_ADJUSTMENT ≥ 0.45`, `place_padding = 1` → 727–760 kµm observed. The 1.05–1.41 M outliers (iter_12, 16, 20, 23, 102, 106) are pure congestion/wire penalty adding up to +0.12 W with no timing benefit; a WL cap would prune them automatically.
6. **Add routed wire length and a congestion/overflow proxy as explicit objectives or constraints.** Power's residual variance is entirely WL-driven, and proxy overflow (430 vs 570) already separates good from bad runs before routing.
7. **Constrain utilization and validate density analytically.** Cap util at 67, require `CORE_MARGIN ≥ 3` above util 65, and precompute global-placement density ≤0.94 to eliminate the `FLW-0024` class entirely (2/147 today, 100 % of hard failures).
8. **Run a replication study (≥3 seeds / re-runs) at the three anchors** (iter_60, iter_88, iter_135). With no repeats, the observed ±0.013 ns / ±250 µm² parent-child deltas cannot be separated from parameter effects — every ranking above is single-sample.
9. **Attack the structural critical path, not the flow knobs.** Best WNS plateaus at −87 ps with `set_max_fanout 10` forcing heavy buffering in the 64-MAC array and the divider. Run `report_checks -path_delay max` on iter_60 and iter_88 to confirm the DCT MAC / `div_su` path, then evaluate RTL-level options (extra MAC pipeline stage, retimed divider, registered zigzag crossbar, duplicate high-fanout `ena`/`dstrb` nets) — this is the only route to WNS ≥ 0 at 1 ns.
10. **Improve promotion policy.** Stop re-promoting exact duplicates (5 of 40 full runs are verbatim re-evaluations) and adopt bias-corrected multi-fidelity (learn the CTS→final offset per objective) plus a minimum predicted-HV-delta threshold. With 90 of 147 trials adding <0.03 % HV, reallocating those runs to steps 3, 4 and 8 is the highest-leverage change to the campaign.

## Appendix: Top Pareto Points

All 10 full-fidelity Pareto points, sorted by WNS (best first). Metrics from `6_report.json`; WL from `orfs.log`.

| # | Run | WNS (ns) | Area (µm²) | Power (W) | WL (µm) | Util | Aspect | ABC_AREA | gp_TD | gp_RD | Pad addon | RLA | CTS (size/diam) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | iter_60 | −0.0867 | 90 836.9 | 0.5553 | 863 430 | 65.98 | 0.971 | 1 | 1 | 1 | 0.127 | 0.594 | 37 / 68 |
| 2 | iter_71 | −0.0953 | 90 700.7 | 0.5570 | 936 301 | 59.23 | 1.165 | 1 | 1 | 1 | 0.220 | 0.451 | 27 / 80 |
| 3 | iter_145 | −0.0954 | 90 728.6 | 0.5530 | 896 293 | 66.00 | 1.080 | 1 | 1 | 0 | 0.200 | 0.480 | 24 / 60 |
| 4 | iter_137 | −0.1690 | 85 856.6 | 0.4909 | 749 709 | 67.39 | 1.193 | 0 | 1 | 0 | 0.223 | 0.482 | 14 / 141 |
| 5 | iter_88 | −0.1875 | 85 495.1 | 0.4887 | 757 743 | 67.31 | 0.875 | 0 | 1 | 0 | 0.156 | 0.556 | 31 / 149 |
| 6 | iter_135 | −0.1902 | 85 901.0 | 0.4878 | 727 622 | 68.50 | 0.820 | 0 | 1 | 1 | 0.240 | 0.550 | 10 / 50 |
| 7 | iter_94 | −0.1939 | 85 436.0 | 0.4882 | 786 662 | 55.00 | 1.180 | 0 | 1 | 1 | 0.080 | 0.360 | 30 / 120 |
| 8 | iter_27 | −0.1940 | 85 520.1 | 0.4875 | 765 099 | 64.00 | 1.050 | 0 | 1 | 0 | 0.230 | 0.420 | 24 / 100 |
| 9 | iter_106 | −0.2752 | 84 212.4 | 0.4800 | 1 048 634 | 66.50 | 0.820 | 0 | 0 | 1 | 0.055 | 0.565 | 26 / 130 |
| 10 | iter_102 | −0.3387 | 84 181.3 | 0.4878 | 1 079 622 | 65.05 | 0.994 | 0 | 0 | 1 | 0.176 | 0.385 | 36 / 97 |

Reference non-Pareto extremes: best partial-stage WNS iter_38 (−0.0822 ns, 90 844.1 µm², 0.5557 W, from `4_1_cts.json` — not full-valid); best partial-stage area iter_18 (83 441.0 µm², −0.396 ns, 0.4765 W); best partial-stage power iter_21 (0.4743 W, 83 802 µm², −0.278 ns). All three use post-CTS metrics and should not be compared directly against the table above.
