# Final DSE Report

## Executive Summary
This run is operationally healthy but not timing-closed. All 130 jobs succeeded and all 130 are usable, but only 45 are full-fidelity/full-valid; 85 are quick/partial CTS-stage points. The full-valid Pareto set has 9 points, with WNS still negative everywhere. Best full-valid WNS is **-24.7801** (`iter_65`), best full-valid area is **2420.27 um²** (`iter_93`), and best full-valid power is **0.0386518 W** (`iter_115`). The quick/proxy bests are better on paper (`iter_109` WNS -24.1148, `iter_64` area 2407.87, `iter_121` power 0.0371823) but are partial CTS metrics and should not be treated as final QoR.

The Pareto frontier shows a clear **power vs WNS** trade-off, with area nearly insensitive: Pareto WNS spans **-24.78 to -45.18**, area spans **2420.27 to 2501.53 um²** (only ~3.4%), and power spans **0.03865 to 0.05998 W** (~55%). Power is bimodal and strongly associated with `ABC_AREA`: `ABC_AREA=1` gives ~0.039–0.040 W, while `ABC_AREA=0` gives ~0.058–0.060 W. Recommended next work is to separate low-power and timing branches, re-rank only full-valid metrics, and run targeted full-fidelity confirmation around the Pareto anchors.

## Search Quality and Data Quality
**Search quality.** The flow is robust: 130/130 job success, 0 hard failures, 130/130 usable, 45 full-valid, and 9 Pareto points. The search explored wide ranges of utilization, aspect ratio, density, CTS clustering, and routing-layer adjustment, and it found a coherent trade-off set. However, optimization quality is only moderate: no point reaches positive WNS, the full-valid best is still -24.78 reported units, and many full runs are far worse (-50 to -183). Area is almost invariant across the Pareto set, so most DSE effort is effectively trading WNS against power. All Pareto full runs are marked `qor_regressed: true` versus their parents, which suggests weak quick-to-full transfer or overly sensitive parent comparisons.

**Data quality.** Metrics are mixed-fidelity. Quick runs use `4_1_cts.json` and are marked `partial`; full runs use `6_report.json` and are marked `full`. The headline `best_wns`, `best_area`, and `best_power` entries are all quick/partial CTS results, so they are not directly comparable with the full-valid Pareto set. In addition, `clock_period` and `ecp` are `Infinity` for essentially all points, quick `wl` is `Infinity`, and `cum_hv`/`cum_hv_delta` are zero in the Pareto metadata, so frequency-normalized QoR and hypervolume progress cannot be audited from this JSON. There is also a count inconsistency: summary reports `full_valid=45`, while `pareto_meta.num_full_valid=41`.

## Pareto Frontier Insights
The Pareto set splits into two families rather than one continuous frontier:

| Family | Pareto runs | WNS range | Area range | Power range | Interpretation |
|---|---|---:|---:|---:|---|
| High-power / timing-oriented, `ABC_AREA=0` | `iter_40`, `iter_72`, `iter_93`, `iter_99` | -25.54 to -37.37 | 2420.27–2439.00 | 0.05846–0.05998 | Better area and generally better WNS, but ~50% higher power |
| Low-power, `ABC_AREA=1` | `iter_25`, `iter_52`, `iter_65`, `iter_74`, `iter_115` | -24.78 to -45.18 | 2465.35–2501.53 | 0.03865–0.04045 | Much lower power, but generally larger area and a wider WNS spread |

Key trade-offs:

- **Power is the main axis.** Moving from the high-power family to the low-power family cuts power by roughly 0.019 W, but usually costs WNS and area. The best low-power WNS point (`iter_65`, -24.78) pays for that timing with the largest Pareto area (2501.53) and slightly higher low-power-family power (0.04045).
- **Area has little room.** The best high-power area point (`iter_93`, 2420.27) and best WNS high-power point (`iter_72`, 2439.00) are close in area, but `iter_72` has higher power and ~12 better WNS. Within the low-power family, `iter_115` is smallest and lowest power but has the worst WNS (-45.18).
- **WNS vs power is not monotonic.** `iter_72` has the best high-power WNS (-25.54) at maximum high-power-family power (0.05998), while `iter_93` reduces area and power slightly but loses ~11.8 WNS. In the low-power family, `iter_74` is a balanced point (-35.09, 2498.21, 0.03998), while `iter_115` minimizes power but sacrifices ~10 WNS versus `iter_74`.

## Parameter Sensitivity
These are observational sensitivities, not controlled causal effects, because parameters are correlated across the sampled points.

| Parameter | Apparent effect | Evidence / guidance |
|---|---|---|
| `ABC_AREA` | Strongest power lever; categorical branch | `ABC_AREA=1` points are ~0.0387–0.0404 W; `ABC_AREA=0` points are ~0.0585–0.0600 W. Timing effect is mixed: `iter_65` is best WNS with `ABC_AREA=1`, but `iter_72` is best high-power WNS with `ABC_AREA=0`. Treat as two search branches. |
| `CORE_UTILIZATION` | Strong non-linear effect; very low utilization is bad | Pareto uses 45.8–54.8%. Full runs below ~42% often have poor WNS and larger area (`iter_127` util 32.5, WNS -67.49, area 2528.27; `iter_126` util 34.0, WNS -61.25). Prefer ~52–55%, with 45.8 only for selected high-power points. |
| `CORE_ASPECT_RATIO` | Moderate; avoid extremes | Pareto uses 0.84–1.12. Extreme aspect ratios appear with poor WNS (`iter_90` 1.20, WNS -49.97; `iter_122` 1.18, WNS -59.11; `iter_126` 1.18, WNS -61.25). Prefer ~0.85–1.10. |
| `PLACE_DENSITY_LB_ADDON` | Non-monotonic; interactive | Pareto uses 0.024–0.248. Low values appear in low-power points (`iter_74` 0.0244), high values appear in best-WNS `iter_72` (0.2424), and the quick best-power point uses 0.2475. Use as a local tuning knob, not a global direction. |
| `gp_timing_driven` | Very strong; must remain enabled | All Pareto points use `gp_timing_driven=1`. Full runs with `0` are catastrophic: `iter_13` -161.7, `iter_15` -146.0, `iter_16` -112.6, `iter_17` -70.4, `iter_71` -183.1. |
| `gp_routability_driven` | Recommended; all Pareto use it | All 9 Pareto points use `gp_routability_driven=1`. Many non-Pareto full runs with `0` remain slower and/or more congested. |
| `place_padding` | Strong consistency in final Pareto | All 9 Pareto points use `place_padding=1`. Quick best-power `iter_121` uses 0, but it is not full-valid. Keep padding on for final points. |
| `CTS_CLUSTER_SIZE` | Pareto consistently 30–40 | All Pareto points use 30–40. Very small clusters appear in poor full runs (`iter_127`/`iter_126` size 12, `iter_82` 17, `iter_76` 18). Avoid small clusters unless deliberately testing. |
| `TNS_END_PERCENT` | Weak / confounded | High values 93–100 appear in both high-power and low-power points; low values 22–55 appear in low-power points. No clean monotonic effect after accounting for `ABC_AREA`. |
| `ROUTING_LAYER_ADJUSTMENT` | Weak / congestive interaction | Pareto range 0.328–0.589. Low values appear in low-power `iter_115` but also high-power `iter_99`; high values appear in `iter_65` best WNS and `iter_74` low power. Tune locally with congestion checks. |
| `CORE_MARGIN` | Weak | Pareto uses 2–7. High margin with low utilization is often bad (`iter_71` margin 7, WNS -183), but `iter_115` uses 7 and is low power. Prefer 2–4 for robust timing. |

## Failure Analysis
There are **no hard failure snippets** in the JSON: `job_fail=0`, `failure_category='ok'`, `fail_stage` is empty, and `fail_log_tail` is empty for the Pareto points. The available evidence is structured symptom summaries rather than tool crash logs. The common near-failure modes are:

| Mode | Evidence | Diagnosis |
|---|---|---|
| Timing setup not closed | `symptoms: ["setup"]` on `iter_25`, `iter_52`, `iter_65`, `iter_74`, `iter_115`; all WNS negative | The flow finishes, but no sampled configuration closes timing. This is the dominant QoR failure. The design is likely limited by architecture/constraints, not placement knobs alone. |
| Congestion / overflow co-occurs with setup | `symptoms: ["overflow","congestion","setup"]` on `iter_40`, `iter_72`, `iter_93`, `iter_99`; many quick runs have proxy overflow 410–1820 | Congestion does not always prevent completion, but high-util/high-density points can trade routability for timing. Use routability-driven placement and padding, and reject high-overflow quick points early. |
| Quick-to-full regression | All Pareto full runs have `qor_regressed: true`; e.g. `iter_65` improves WNS by 9.90 but adds 24.15 area and 0.00136 W | Quick CTS ranking is not a reliable final selector. Full-route/final-report confirmation is required before declaring improvements. |
| Disabling timing-driven placement | `gp_timing_driven=0` full runs: `iter_13`, `iter_15`, `iter_16`, `iter_17`, `iter_71` | Catastrophic WNS (-70 to -183). This is a hard “do not use” region for this design. |
| Extreme low utilization / high margin / small CTS clusters | `iter_126`, `iter_127`, `iter_20`, `iter_71` | Long wirelength, larger area, high power, or severe WNS. These are configuration failure modes rather than flow crashes. |

## Recommended Next Experiments
1. **Clean the result table before further search.** Exclude quick/partial points from “best” reporting, recompute Pareto only on true full-valid metrics, record `clock_period`, ECP, and units, and resolve the 45 vs 41 full-valid count discrepancy. `cum_hv` should also be populated or removed.
2. **Run full-fidelity confirmation around the four key Pareto anchors:** `iter_65` (best WNS), `iter_72` (best high-power WNS), `iter_93` (best area), and `iter_115` (best power). Use 2–3 replicates or small perturbations to estimate noise before trusting close comparisons.
3. **Split the search by `ABC_AREA`.** For low power, constrain to `ABC_AREA=1`, target power <0.040 W, and sample utilization 52–55%, aspect 0.88–1.00, `place_padding=1`, CTS cluster size 30–40. For timing/area, use `ABC_AREA=0` with utilization 52–55%, aspect 0.88–1.10, and density 0.16–0.24.
4. **Prune known-bad regions.** Avoid `gp_timing_driven=0`, `place_padding=0`, utilization below ~45% unless specifically exploring the 45.8% high-power corner, aspect ratio above ~1.15, and CTS cluster sizes below ~25. These regions account for most poor full-valid points.
5. **Do a timing root-cause check on `iter_65` and `iter_72`.** Report top reg-to-reg and IO paths, and separate decoder/ALU/CSR/LSU paths. If boundary paths dominate, revisit the 20% IO delay assumption; if internal paths dominate, DSE alone will not close the -24 reported-unit gap.
6. **Add targeted placement repair experiments.** For the best full-valid candidates, run `estimate_parasitics -placement` and `repair_design` with modest slew/cap margins, then re-check `report_wns`, `report_tns`, and placement legality. This tests whether post-placement repair can improve WNS without moving to the high-power branch.
7. **Use proxy overflow as an early filter for quick runs.** Reject or deprioritize quick points with proxy route overflow far above the observed low-overflow band (~410–600) unless they are uniquely valuable. This should reduce wasted full promotions.
8. **Run a clock-period diagnostic sweep on the best parameters.** Re-run `iter_65`, `iter_72`, `iter_93`, and `iter_115` at relaxed periods such as 1100, 1250, and 1468 ps. If WNS moves near zero, the current 1000 ps target is the binding issue; if not, focus on architecture and logic restructuring.

## Appendix: Top Pareto Points
Sorted roughly by WNS.

| Run | ABC_AREA | Util | Aspect | Margin | Density addon | RLA | CTS size/diam | WNS | Area | Power | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `iter_65` | 1 | 46.8331 | 0.8846 | 3 | 0.0625 | 0.5776 | 39/25 | -24.7801 | 2501.53 | 0.0404488 | Best full-valid WNS; largest Pareto area |
| `iter_72` | 0 | 53.0000 | 1.1000 | 2 | 0.2424 | 0.4100 | 30/40 | -25.5362 | 2439.00 | 0.0599752 | Best high-power WNS; max power |
| `iter_74` | 1 | 52.3583 | 0.9806 | 2 | 0.0244 | 0.5886 | 35/24 | -35.0940 | 2498.21 | 0.0399832 | Balanced low-power point |
| `iter_40` | 0 | 45.8000 | 0.8400 | 2 | 0.1660 | 0.5280 | 33/60 | -35.1170 | 2433.75 | 0.0585473 | Good area/high-power point |
| `iter_99` | 0 | 54.6175 | 0.9390 | 5 | 0.1804 | 0.3579 | 39/22 | -35.2661 | 2426.52 | 0.0587883 | Near-best area, high power |
| `iter_93` | 0 | 54.8000 | 0.8800 | 6 | 0.1180 | 0.4900 | 38/78 | -37.3718 | 2420.27 | 0.0584628 | Best full-valid area |
| `iter_25` | 1 | 49.8163 | 1.1108 | 4 | 0.1049 | 0.5684 | 36/77 | -40.2594 | 2498.08 | 0.0396560 | Low power, mid WNS |
| `iter_52` | 1 | 45.8000 | 0.8400 | 2 | 0.1660 | 0.5500 | 40/80 | -44.4199 | 2497.28 | 0.0393620 | Lower power, worse WNS |
| `iter_115` | 1 | 54.1026 | 0.8881 | 7 | 0.1164 | 0.3278 | 38/57 | -45.1764 | 2465.35 | 0.0386518 | Best full-valid power; worst Pareto WNS |
