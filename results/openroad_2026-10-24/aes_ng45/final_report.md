# Final DSE Report

## Executive Summary

The campaign is highly feasible but only partially validated. It completed **313/313 successful jobs with 0 failures and 313 usable runs**, but only **99 runs are full-valid** for final Pareto comparison; **214 runs are quick/partial CTS-stage measurements**. The final full-valid Pareto frontier contains **9 points**.

The headline `best_wns`, `best_area`, and `best_power` values in the JSON are **not full-valid final results**. They are sourced from `4_1_cts.json` with missing wirelength and are quick/partial metrics:

- `best_wns`: iter_180, WNS 0.11182, Area 17271.6, Power 0.219498 — partial
- `best_area`: iter_286, Area 17183.6, WNS 0.0897971, Power 0.218416 — partial
- `best_power`: iter_304, Power 0.217908, WNS 0.0861298, Area 17200.4 — partial

Among **full-valid Pareto points**, the real frontier is:
- Best WNS: **iter_78**, WNS **0.0983699**, Area **17219.8**, Power **0.223107**
- Minimum area: **iter_111**, Area **17191.3**, WNS **0.0964068**, Power **0.222918**
- Minimum power: **iter_175**, Power **0.222444**, WNS **0.0943348**, Area **17210.2**

The Pareto frontier is very tight: full-valid area spans only **17191.3–17219.8 um²** and power spans **0.222444–0.223223 W**, while WNS spans **0.07287–0.09837**. WNS is the dominant differentiating objective; area and power differences are small and may be near run-to-run noise.

The most credible operating region is: **CORE_UTILIZATION 54–55, CORE_ASPECT_RATIO ~0.94–1.00, CORE_MARGIN 2–3, PLACE_DENSITY_LB_ADDON ~0.235–0.25, TNS_END_PERCENT ~77–91, cts_cluster_size ~28–35, CTS_CLUSTER_DIAMETER ~110–125, ROUTING_LAYER_ADJUSTMENT ~0.55–0.59, place_padding=0, gp_timing_driven=1, gp_routability_driven=0, ABC_AREA=1**.

There are **no hard flow failures** and no useful `fail_log_tail` snippets. The dominant “failure” modes are soft: pervasive congestion/overflow warnings and systematic quick-to-full QoR regression.

## Search Quality and Data Quality

### Search Quality

- **Feasibility is excellent:** 313/313 jobs succeeded, 0 job failures, 0 unusable runs. The design closes across a wide parameter range.
- **Final comparability is limited:** only 99/313 runs, or **31.6%**, are full-valid; 214/313 are partial quick metrics. Pareto analysis therefore uses 99 points, not 313.
- **Pareto yield:** 9/99 full-valid points are Pareto, giving a reasonably rich but narrow final frontier.
- **Convergence signal:** late-run hypervolume gains are effectively zero. For example, `cum_hv_delta` is 0.002 for iter_227, 0.001 for iter_271, and 0.0 for iter_312; EHVI scores late in the campaign are near `1e-9`. The search has largely saturated the promoted/full-valid region.
- **Possible premature promotion saturation:** several attractive quick points remain unvalidated, including the JSON headline `best_wns`, `best_area`, and `best_power` points. The search may have converged on the full-valid pool while leaving promising quick candidates unpromoted.
- **Sampling is concentrated in a good region:** all full-valid Pareto points are clustered at `CORE_UTILIZATION` 54.0–54.97 and `PLACE_DENSITY_LB_ADDON` 0.233–0.249. This is consistent with a mature local exploitation phase.

### Data Quality

- **Job success vs metric validity must be separated:** all 313 runs are “usable,” but only 99 have full final metrics. Quick points are useful for screening, not final Pareto claims.
- **Metric sources differ:** quick/partial metrics come from `4_1_cts.json`; full-valid metrics come from `6_report.json`. Full-valid points also have wirelength from `orfs.log`; quick points have `wl = Infinity`.
- **ECP is unusable:** `clock_period` is null/Infinity for all reported points, so derived ECP is Infinity. Do not use ECP for ranking.
- **Quick-to-full bias is systematic:** promoted full points show area unchanged versus parent, but WNS regresses by roughly **−0.001 to −0.0275** and power increases by roughly **+0.0044 to +0.0048 W** in `qor_vs_parent`. Quick results are optimistic.
- **No failure snippets exist:** all Pareto points have empty `fail_log_tail` and `failure_category: "ok"`. Failure analysis must rely on `log_summary.symptoms` and QoR regression signals.
- **Congestion symptoms are nearly universal but not fatal:** the dominant symptom string is `["overflow", "congestion"]`. This is a routability stress indicator, not a job failure.

## Pareto Frontier Insights

All 9 full-valid Pareto points share:
`gp_timing_driven=1`, `gp_routability_driven=0`, `place_padding=0`, `ABC_AREA=1`.

Key trade-off corners:

- **WNS-max corner — iter_78:** WNS **0.0983699**, Area **17219.8**, Power **0.223107**, WL 263633. This is the best full-valid WNS. It pays a modest area and power penalty versus the area-min point.
- **Area-min corner — iter_111:** Area **17191.3**, WNS **0.0964068**, Power **0.222918**, WL 263408. This is a strong balanced point: minimum area with WNS only ~0.00196 below the WNS-max point.
- **Power-min corner — iter_175:** Power **0.222444**, WNS **0.0943348**, Area **17210.2**, WL 261980. The power gain versus WNS-max is only ~0.000663 W, while WNS drops by ~0.00404.
- **Low-WNS frontier edge — iter_112:** WNS **0.0728665**, Area **17201.4**, Power **0.222692**, WL 261219. It remains Pareto mainly through low power, but it is unattractive if WNS matters.
- **Lowest wirelength among Pareto — iter_227:** WL **261005**, WNS **0.0864128**, Area **17208.9**, Power **0.222519**. It uses larger CTS clusters (`40/150`) and lower RLA (`0.52`) than the central region.

Overall, the frontier is not a broad area-power trade-off. It is a **WNS-dominated frontier with small area/power perturbations**. Many adjacent Pareto points differ by less than expected run-to-run noise. Therefore, the practical recommendation is to prefer **iter_111** as a balanced full-valid reference and **iter_78** as the WNS-max reference, then validate repeatability before treating fine differences as real.

## Parameter Sensitivity

These are apparent sensitivities from the sampled DSE data, not controlled causal estimates. Several parameters are confounded with each other, especially high aspect ratio, high core margin, low RLA, and routability-driven placement.

| Parameter | Apparent effect on WNS / Area / Power | Evidence from this run | Recommended region |
|---|---|---|---|
| `CORE_UTILIZATION` | Strongest apparent knob. Higher utilization near 54–55 improves area and power while maintaining WNS; going too low worsens area and often WNS. | All 9 Pareto points are 54.0–54.97. Runs near 51.4, 50, 46.5, 40 are generally worse in area/WNS. | **54–55**; test up to 55.5 only with full validation |
| `PLACE_DENSITY_LB_ADDON` | Strong effect. Low values hurt WNS and power; very low values are catastrophic. | Pareto range 0.233–0.249. iter_140 at 0.02: WNS 0.0616, Power 0.2280, Area 17363.7. iter_246 at 0.08: WNS 0.0709, Power 0.2253. | **0.235–0.25**; avoid below ~0.20 |
| `ABC_AREA` | Mandatory. `ABC_AREA=0` collapses area and power. | iter_264 (`ABC_AREA=0`, `gp_timing_driven=0`): Area 19243.2, WNS 0.000122, Power 0.301742. iter_284: WNS −0.01785, Power 0.307223. | **1** |
| `gp_timing_driven` | Mandatory in practice. Turning it off with `ABC_AREA=0` caused severe QoR collapse. | All Pareto points use 1. iter_264/284 use 0 and are far outside the good region. | **1** |
| `place_padding` | Prior evidence and all Pareto points say keep 0. | All 9 Pareto points have 0. No controlled variation in this frontier, but design prior strongly supports 0. | **0** |
| `CORE_ASPECT_RATIO` | Near-square is safer. Extremes 1.15–1.2 with large margin/low RLA hurt WNS and area. | iter_238 (`AR=1.2`, `CM=8`, `RLA=0.3`): WNS 0.0801, Area 17301.4, Power 0.2243. iter_290 (`AR=1.15`): WNS 0.0713. Pareto mostly 0.94–1.00. | **0.93–1.00** preferred; 0.82–0.90 only with validation |
| `CORE_MARGIN` | Low-to-moderate margin is better. High margin often appears with worse WNS. | Pareto uses 1–4, with 2–3 common. High margin 6–8 appears in poor full points such as iter_238, iter_290, iter_140. | **2–3** |
| `TNS_END_PERCENT` | Low values hurt WNS. | Pareto uses 77–91. iter_140 at 10: WNS 0.0616. iter_246 at 20: WNS 0.0709. | **75–90**; avoid ≤40 for final candidates |
| `ROUTING_LAYER_ADJUSTMENT` | Central-high range is safer. Very low RLA with high aspect/margin causes poor full results. | Pareto uses 0.52–0.592. Very low 0.3 points have poor WNS/area. | **0.55–0.59**; avoid <0.50 unless deliberately stress-testing |
| `cts_cluster_size` / `CTS_CLUSTER_DIAMETER` | Weak second-order effect. Smaller clusters appear slightly better for WNS; large clusters can still be Pareto. | Pareto size 24–40, diameter 102–150. Best WNS uses 30/102. iter_227 uses 40/150 but lower WNS. | **28–35 / 110–125** as default |
| `gp_routability_driven` | Not favored by the Pareto set. Enabled runs are overrepresented among high-overflow, poor-QoR full points. | All Pareto points use 0. iter_238, iter_246, iter_290 use 1 and have poor WNS or power. | **0** unless explicitly targeting routing stress |

Power is especially flat on the full-valid frontier: 0.222444–0.223223 W. The large quick-metric power advantage of iter_304 (0.217908 W) is **not** reflected in full-valid results and should not be trusted until fully promoted.

## Failure Analysis

There are no hard failures to quote: all full Pareto entries have `fail_log_tail: ""`, `failure_category: "ok"`, and `success: true`. The meaningful failure modes are soft.

### 1. Congestion and overflow warnings are systemic

Nearly every run reports `symptoms: ["overflow", "congestion"]`, with no flow failure. The `proxy_route_overflow` values in quick runs range from ~380 to ~1560. Higher values correlate with poor full QoR when promoted.

Examples:
- iter_236: quick, `proxy_route_overflow=1560`, WNS 0.07428, with low `PLACE_DENSITY_LB_ADDON=0.06`.
- iter_241: quick, overflow 1030, WNS 0.08469; its full counterpart iter_246 regresses to WNS 0.07085 and Power 0.225319.
- iter_238: full, overflow/congestion symptoms, extreme `AR=1.2`, `CM=8`, `RLA=0.3`; WNS 0.08012, Area 17301.4, Power 0.224305.
- iter_290: full, `AR=1.15`, `CM=6`, `gp_routability_driven=1`; WNS 0.07131, Power 0.223712, WL 267492.

Diagnosis: these are routability/congestion stress cases, not tool crashes. They should be filtered or penalized during promotion.

### 2. Quick-to-full regression is the main “validation failure”

Every promoted point has `qor_regressed: true`. The full run is consistently worse than the quick parent in WNS and power:
- iter_112: `qor_vs_parent` WNS −0.0162138, power +0.004598.
- iter_270: WNS −0.0255584, power +0.004673.
- iter_282: WNS −0.0275253, power +0.004631.

Area is unchanged in these comparisons. Diagnosis: CTS-stage quick metrics are optimistic. Promotion must use a calibrated quick-to-full correction, not raw quick WNS/power.

### 3. Silent QoR collapse from extreme configuration switches

- iter_264: `ABC_AREA=0`, `gp_timing_driven=0`, no overflow symptoms, but Area 19243.2, WNS 0.000122, Power 0.301742. This is a “silent” catastrophic quality loss with no failure string.
- iter_284: `ABC_AREA=0`, `gp_routability_driven=1`, symptoms include `"setup"`, WNS −0.017847, Power 0.307223, Area 19358.4. This is the closest thing to a functional QoR failure, but it is still a successful flow run.

Diagnosis: `ABC_AREA=1` and `gp_timing_driven=1` should be hard constraints for production candidates. `ABC_AREA=0` is useful only as a negative control.

### 4. Extreme aspect/margin/low-RLA combinations waste runs

iter_140 (`CU=45`, `AR=1.2`, `CM=8`, `PLBA=0.02`, `TNS=10`, `RLA=0.3`) produced WNS 0.061567, Power 0.227988, Area 17363.7, WL 297625. This is not a failure, but it is far outside the useful frontier. Diagnosis: avoid combined extremes; one-factor boundary tests are cheaper.

## Recommended Next Experiments

1. **Fully validate the unpromoted quick leaders first.** Highest priority: iter_180 (quick WNS 0.11182), iter_308 (0.105529), iter_255 (0.104705), iter_283 (0.104116), iter_280 (0.102944), iter_245 (0.102854), iter_257 (0.102178), iter_303 (0.102332), iter_286 (quick min area 17183.6), and iter_304 (quick min power 0.217908). These are not final Pareto points until full-valid. Expect WNS regression based on the iter_247→iter_282 and similar pairs.

2. **Run a local exploitation batch around the full-valid Pareto region.** Use `CORE_UTILIZATION` 54.0–55.0 in small steps, `CORE_ASPECT_RATIO` 0.94–1.00, `CORE_MARGIN` 2–3, `PLACE_DENSITY_LB_ADDON` 0.235–0.250, `TNS_END_PERCENT` 77–90, `cts_cluster_size` 28–35, `CTS_CLUSTER_DIAMETER` 110–125, `ROUTING_LAYER_ADJUSTMENT` 0.55–0.59, with `place_padding=0`, `gp_timing_driven=1`, `gp_routability_driven=0`, `ABC_AREA=1`.

3. **Test the utilization cap explicitly.** Run a one-factor `CORE_UTILIZATION` sweep at 54.0, 54.5, 55.0, 55.25, and 55.5 while holding the rest at a balanced Pareto configuration such as iter_111. Do not push beyond 55.5 without full validation.

4. **Repeat the best Pareto points 2–3 times.** The frontier’s power range is only ~0.00078 W and area range only ~28.5 um². Treat very small differences as noise until repeated. A practical screening threshold is to require a meaningful WNS gain or a clearly lower area/power before replacing the current balanced point.

5. **Calibrate quick-to-full deltas.** Fit a simple correction or lower-confidence bound for WNS and power versus the quick parent. Promotion scoring should penalize predicted WNS regression and power increase, not just raw quick EHVI. This directly addresses the observed `qor_regressed: true` pattern.

6. **Use congestion-aware promotion.** Add a hard or soft filter on `proxy_route_overflow` for quick runs. Avoid promoting points with overflow ≳800–1000 unless the configuration is otherwise highly valuable. Keep `gp_routability_driven=0` as the default; use `1` only in a controlled routing-stress study.

7. **Run controlled one-factor sensitivity tests for confounding knobs.** In particular, sweep `PLACE_DENSITY_LB_ADDON` at 0.20, 0.225, 0.235, 0.245, 0.25; `CORE_MARGIN` at 1, 2, 3, 4; `CORE_ASPECT_RATIO` at 0.90, 0.95, 1.00; and `ROUTING_LAYER_ADJUSTMENT` at 0.52, 0.55, 0.58, 0.60. This will separate true effects from combined-extreme sampling.

8. **Stop investing in macro placement for this design.** This is an AES standard-cell/control-dominated design with no hard macros in the parameter set. `rtl_macro_placer` and `macro_placement -style corner_max_wl` are general-flow guidance only and have no validated role here.

9. **Fix metric instrumentation before the next campaign.** Capture `clock_period` so ECP is finite, ensure wirelength is recorded for quick runs if possible, and keep full-valid and partial metrics in separate ranking tables. The current `Infinity` ECP and missing quick wirelength limit secondary analysis.

10. **If power is the primary objective, treat it as a guarded optimization.** Full-valid power is nearly flat across the frontier. The apparent quick power winner, iter_304, may not survive full validation. Optimize power only under WNS/area guardrails and validate with repeats.

## Appendix: Top Pareto Points

All rows are **full-valid Pareto points**. Common settings: `gp_timing_driven=1`, `gp_routability_driven=0`, `place_padding=0`, `ABC_AREA=1`.

| Run | CU | AR | CM | PLBA | TNS | CTS size/diam | RLA | Area | WNS | Power | WL | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iter_78 | 54.0 | 0.92 | 2 | 0.244 | 91 | 30/102 | 0.566 | 17219.8 | 0.0983699 | 0.223107 | 263633 | Best full-valid WNS |
| iter_92 | 54.603 | 0.8289 | 3 | 0.2442 | 77 | 24/105 | 0.5921 | 17210.2 | 0.0968243 | 0.223223 | 265848 | Low aspect, high power edge |
| iter_111 | 54.1499 | 0.9756 | 3 | 0.2473 | 80 | 30/116 | 0.5689 | 17191.3 | 0.0964068 | 0.222918 | 263408 | Minimum area, balanced |
| iter_175 | 54.9718 | 0.987 | 2 | 0.2479 | 85 | 31/119 | 0.5822 | 17210.2 | 0.0943348 | 0.222444 | 261980 | Minimum power |
| iter_312 | 54.259 | 0.984 | 3 | 0.2333 | 77 | 31/122 | 0.5834 | 17203.0 | 0.0902426 | 0.222915 | 264326 | Late stable point |
| iter_271 | 54.5265 | 0.9997 | 3 | 0.2435 | 85 | 32/123 | 0.5542 | 17200.1 | 0.0895491 | 0.222821 | 262892 | Near-square aspect |
| iter_227 | 54.9 | 0.96 | 4 | 0.245 | 85 | 40/150 | 0.52 | 17208.9 | 0.0864128 | 0.222519 | 261005 | Lowest Pareto WL, larger CTS |
| iter_149 | 54.2 | 0.9432 | 2 | 0.2485 | 77 | 30/123 | 0.5815 | 17201.2 | 0.0834761 | 0.222715 | 261262 | Good area/power balance |
| iter_112 | 54.85 | 0.96 | 1 | 0.2464 | 89 | 35/116 | 0.5663 | 17201.4 | 0.0728665 | 0.222692 | 261219 | Low-WNS frontier edge |
