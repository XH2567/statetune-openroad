# Final DSE Report

## Executive Summary

- The run is operationally robust: 226/226 jobs succeeded, 0 failures, and 226/226 points are marked usable.
- Only 73/226 points are full-valid, however. The remaining 153 are quick/CTS-only partials, so headline best-area/best-power values must not be treated as final routed QoR.
- Best full-valid WNS is iter_51: WNS +2.803, Area 1631.49, Power 0.149329, WL 81236, at CORE_UTILIZATION 63.14.
- The full-valid Pareto front is timing-skewed: only iter_51 has positive WNS. Low-area and low-power full-valid points sacrifice very large WNS, often reaching -13 to -21.
- The JSON summary’s best-area and best-power point is iter_224, but it is a quick partial CTS result with WL = Infinity and WNS -5.559. It is not full-valid and is not in the full Pareto set.
- Hypervolume is effectively saturated after iter_51: cumulative HV reaches about 310.62M at iter_51 and ends at about 310.75M, a gain of only ~0.04% over the remaining run. Later Pareto additions mostly extend low-area/low-power niches rather than improve timing.
- Main timing levers in this data: gp_timing_driven=1, ABC_AREA=1, CORE_UTILIZATION near 63–65%, high TNS_END_PERCENT, moderate PLACE_DENSITY_LB_ADDON near 0.10–0.12, high ROUTING_LAYER_ADJUSTMENT around 0.55–0.59, and gp_routability_driven=0 for timing-first runs.

## Search Quality and Data Quality

**Search quality**

- Execution quality is excellent: 226/226 job_success and 226/226 usable, with no failure category to diagnose.
- Valid comparison quality is moderate: 73/226 full_valid = 32.3%. The Pareto set contains 17 full-valid points, i.e. 23.3% of the full-valid set.
- The search found the main WNS winner early. iter_51 accounts for essentially all final hypervolume; later EHVI gains are small and often produce Pareto points with much worse WNS.
- The Pareto front is broad but highly unbalanced: Area spans roughly 1606.75–1675.65, Power roughly 0.13902–0.14936, while WNS spans -21.17 to +2.80. This is a real multi-objective trade-off, not a tight near-optimal frontier.
- Most full-valid runs still have negative WNS. The search is robust but has not produced many near-timing-clean alternatives to iter_51.

**Data quality**

- job_success is not the same as full_valid here: all runs succeed, but 153/226 are partial quick metrics from 4_1_cts.json, with no final routed WL and only proxy_route_overflow.
- The best_area and best_power summary records are both iter_224, a quick partial point: Area 1606.5, Power 0.135345, WNS -5.55925, WL Infinity. This is useful as a low-area/low-power candidate, but it is not a final Pareto point.
- Full-valid metrics are sourced from 6_report.json and full WL from orfs.log. Full Pareto points have finite WL; quick partial points generally have WL = Infinity.
- clock_period and derived ECP are Infinity throughout because clock_period is null. ECP is therefore unusable for analysis.
- The WNS unit is not conclusively resolved from the data. Values are consistent within this platform, but physical interpretation of WNS magnitudes should be treated cautiously.
- qor_vs_parent is often “regressed”, but many comparisons are between a quick parent and a full child, so this should not be read as a clean failure signal.
- There are no failure snippets to inspect: fail_log_tail is empty for the provided records. The available diagnostic evidence is log_summary symptom tags, which are non-fatal.

## Pareto Frontier Insights

Representative full-valid Pareto points:

| Point | WNS | Area | Power | WL | Util | Key setting | Interpretation |
|---|---:|---:|---:|---:|---:|---|---|
| iter_51 | +2.803 | 1631.49 | 0.14933 | 81236 | 63.14 | timing=1, rout=0, ABC=1, TNS=99, route_adj=0.584 | Best timing; higher power |
| iter_56 | -1.905 | 1627.92 | 0.14936 | 77176 | 65.74 | timing=1, rout=1, ABC=1, TNS=40 | Near-timing alternative, but high power |
| iter_187 | -2.168 | 1617.64 | 0.14773 | 76726 | 63.20 | timing=1, ABC=1, TNS=95, route_adj=0.550 | Attractive WNS-area compromise |
| iter_208 | -3.262 | 1675.65 | 0.14115 | 90665 | 66.00 | aspect=1.2, padding=1 | Lower power, high area, moderate WNS loss |
| iter_155 | -4.219 | 1648.46 | 0.13950 | 85976 | 58.50 | timing=1, ABC=0, TNS=10 | Power-saving point; area penalty and WNS loss |
| iter_75 | -5.959 | 1646.24 | 0.13904 | 85449 | 67.85 | timing=1, rout=1, ABC=0 | Good power, but timing and area compromised |
| iter_149 | -17.319 | 1606.75 | 0.13951 | 97994 | 66.93 | timing=0, ABC=0, TNS=16, route_adj=0.310 | Minimum full-valid area, but severe timing loss |
| iter_96 | -21.170 | 1608.99 | 0.13902 | 107535 | 68.62 | timing=0, ABC=0, TNS=10 | Minimum full-valid power, worst timing |

Trade-off interpretation:

- The timing-first frontier is anchored by iter_51 at WNS +2.803 and iter_187 at WNS -2.168. Moving from iter_51 to iter_187 saves about 13.85 area units and 0.00160 W, but gives up about 4.97 WNS.
- Low-power full-valid points around 0.1390–0.1395 W require accepting WNS in the -4 to -21 range. The best power saving relative to iter_51 is about 0.0103 W, at the cost of 7 to 24 WNS units.
- Low-area full-valid points are clustered around 1606.75–1610.6, but are timing-poor. The area gain relative to iter_51 is only about 21–25 µm², while WNS can degrade by 10–24 units.
- iter_208 is an interesting low-power, moderate-WNS point, but its area is high at 1675.65 due to aspect ratio 1.2 and place_padding=1. It is not a compact solution.
- The design-context recommendation of roughly 60–66% utilization for timing-first runs is consistent with the data. The full Pareto also contains 51–58% and 67–70% points, but those are driven by area/power objectives and are mostly timing-poor.

## Parameter Sensitivity

These are observational associations, not clean causal estimates, because the search is non-orthogonal.

| Parameter | Apparent WNS effect | Apparent Area effect | Apparent Power effect | Evidence |
|---|---|---|---|---|
| CORE_UTILIZATION | Best near 63.14; >67 often much worse | Higher utilization tends to reduce area | Higher utilization can reduce power, but confounded | iter_51 at 63.14 is best WNS; iter_96, iter_149, iter_215, iter_225 at 66.9–69.8 are timing-poor |
| CORE_ASPECT_RATIO | Near 1.0–1.08 best; very low/high mixed | High aspect, e.g. 1.2, increases area | Weak | iter_51 aspect 1.065; iter_208 aspect 1.2 has area 1675.65 |
| PLACE_DENSITY_LB_ADDON | Lower around 0.10–0.12 favorable; >0.20 often bad | Not monotonic | Weak | iter_51 uses 0.114; several severe WNS points use 0.20–0.23 |
| gp_timing_driven | Strong favorable for WNS | Often increases area/power | Often increases power | Many best/near-best WNS points use 1; many severe WNS points use 0 |
| gp_routability_driven | Worsens WNS | Tends to lower area | Mixed | Full Pareto routability-on points iter_215, iter_126, iter_23 have poor WNS |
| ABC_AREA | 1 strongly favorable for WNS and area | Tends to lower area | Increases power | Best WNS iter_51 uses 1; low-power full points often use 0 |
| TNS_END_PERCENT | High 90–99 favorable | Weak | Weak | iter_51 uses 99, iter_187 uses 95; low values 10–16 often appear in severe WNS points |
| ROUTING_LAYER_ADJUSTMENT | High 0.55–0.59 favorable | Weak | Weak | iter_51 0.584, iter_187 0.550, iter_208 0.600; many low-0.30 points are timing-poor |
| cts_cluster_size / CTS_CLUSTER_DIAMETER | Best WNS point uses 18/41 | Larger clusters often reduce area | Larger clusters can increase power | iter_51 18/41; high cluster-size points include both low-area and poor-WNS cases |
| place_padding / CORE_MARGIN | Best uses padding=0, margin=6 | padding=1 appears in several high-area points | Weak | iter_208 padding=1 and aspect=1.2 gives high area |

Practical interpretation:

- For a timing-first run, the strongest recipe is close to iter_51: utilization ~63–65%, aspect ~1.0–1.08, density addon ~0.10–0.12, timing-driven on, routability-driven off, ABC_AREA=1, TNS_END_PERCENT high, routing layer adjustment ~0.55–0.59.
- For power-first runs, ABC_AREA=0 and timing-driven=0 are common, but they typically lose several WNS units. They should be treated as separate power-oriented experiments, not as substitutes for the timing winner.
- High utilization above ~67% can reduce area/power, but in this data it is repeatedly associated with severe timing loss. Do not extend beyond ~66% without checking S-box/MixColumns routing hotspots.

## Failure Analysis

There are no job failures and no populated failure snippets. fail_log_tail is empty, failure_category is “ok”, and job_fail = 0. The failure analysis therefore concerns non-fatal QoR and log symptoms, not tool crashes.

| Symptom | Evidence | Likely interpretation | Action |
|---|---|---|---|
| setup | Present in nearly all full-valid log_summary records; negative WNS appears in most full Pareto points | Timing closure is the dominant QoR problem, not tool failure | Focus on timing-driven, ABC_AREA, TNS_END_PERCENT, routing adjustment, and utilization; do not count these as job failures |
| overflow / congestion | Many full and quick records include overflow/congestion tags; quick proxy_route_overflow ranges roughly 350–960 | Routing pressure is common, especially in aggressive utilization/density configurations | Use proxy_route_overflow as a warning flag; keep timing-first utilization near 63–66 and density near 0.10–0.12 |
| large WL with poor WNS | iter_64 WL 119413 WNS -11.891; iter_23 WL 108067 WNS -13.784; iter_96 WL 107535 WNS -21.170 | Congestion/detour likely amplifies timing degradation | Inspect routing hotspots and avoid high-density/low-routing-adjustment combinations |
| qor_regressed=true | Many full records regress versus parent | Often a quick-vs-full fidelity artifact, not a clean failure | Compare only full-valid to full-valid points for true QoR regression |
| “best area/power” from partial metrics | iter_224 is quick, WL Infinity, metrics_status partial | The apparent best area/power is not a final routed result | Promote iter_224-like candidates to full flow before accepting them |

Common bad patterns in the data:

- High CORE_UTILIZATION around 67–70 with gp_timing_driven=0 and/or ABC_AREA=0 often produces very negative WNS, e.g. iter_96, iter_149, iter_215.
- High PLACE_DENSITY_LB_ADDON around 0.20–0.23 combined with low ROUTING_LAYER_ADJUSTMENT around 0.30–0.33 repeatedly appears in severe timing failures.
- gp_routability_driven=1 can lower area but tends to worsen WNS. It is useful for area exploration, not for timing-first closure.
- Low TNS_END_PERCENT values around 10–20 appear in several of the worst WNS points, although the effect is confounded by other parameters.

## Recommended Next Experiments

1. Re-run iter_51 as a constraint-complete baseline. Add the missing set_clock_uncertainty, set_driving_cell, set_load, set_max_transition, false path on async rst, multicycle paths, clock-gating checks, and MCMM corners. The current +2.803 WNS should be treated as provisional until constraints are complete.

2. Run a local DOE around iter_51 rather than broad random exploration: CORE_UTILIZATION 62–66, CORE_ASPECT_RATIO 1.00–1.08, PLACE_DENSITY_LB_ADDON 0.09–0.12, TNS_END_PERCENT 90–99, ROUTING_LAYER_ADJUSTMENT 0.55–0.60, cts_cluster_size 16–22, CTS_CLUSTER_DIAMETER 38–46.

3. Validate low-area/low-power quick candidates in full flow. Promote iter_224 and similar ABC_AREA=0, timing_driven=0, high-utilization points to finish. Do not use their CTS-only area/power as final Pareto data.

4. Run a 2×2 experiment on ABC_AREA and gp_timing_driven: ABC_AREA in {0,1} × timing_driven in {0,1}, holding the iter_51 physical settings. This will separate WNS benefit from the power penalty.

5. Sweep the highest-impact timing knobs explicitly: TNS_END_PERCENT {85, 90, 95, 99}, ROUTING_LAYER_ADJUSTMENT {0.50, 0.55, 0.58, 0.60}, and PLACE_DENSITY_LB_ADDON {0.08, 0.10, 0.11, 0.12}. Keep utilization in the 63–65% range.

6. Keep gp_routability_driven=0 for timing-first runs. Use routability-driven=1 only as a separate area-exploration branch, and always re-check WNS before adding those points to the timing Pareto.

7. Use proxy_route_overflow as a quick-stage warning, not as a final objective. Flag or reject quick candidates with extreme overflow before spending full-flow runtime, but do not equate a non-zero proxy value with job failure.

8. Add a design-level critical-path experiment for the S-box/MixColumns network: LUT-style synthesis, S-box decomposition, and retiming evaluation. Knob tuning is now close to saturated; the dominant remaining timing lever may be logic structure.

9. Stop broad random search after the current HV plateau. Allocate most new runs to local exploitation around iter_51, a smaller block to full-valid low-power validation, and a final block to MCMM/corner and constraint validation.

10. Improve future diagnostics by capturing non-empty failure-stage snippets and per-stage timing/congestion summaries. The current dataset has no fail_log_tail content, which limits root-cause analysis for near-failing configurations.

## Appendix: Top Pareto Points

All rows below are full-valid Pareto points, sorted by WNS descending. Key: Tim = gp_timing_driven, Rout = gp_routability_driven, Dens = PLACE_DENSITY_LB_ADDON, RAdj = ROUTING_LAYER_ADJUSTMENT, Pad/Mar = place_padding/CORE_MARGIN.

| Run | WNS | Area | Power | WL | Util | Aspect | Dens | Tim | Rout | TNS | ABC | CTS size/diam | RAdj | Pad/Mar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| iter_51 | +2.803 | 1631.49 | 0.14933 | 81236 | 63.14 | 1.065 | 0.114 | 1 | 0 | 99 | 1 | 18/41 | 0.584 | 0/6 |
| iter_56 | -1.905 | 1627.92 | 0.14936 | 77176 | 65.74 | 0.874 | 0.094 | 1 | 1 | 40 | 1 | 37/65 | 0.492 | 0/1 |
| iter_187 | -2.168 | 1617.64 | 0.14773 | 76726 | 63.20 | 1.060 | 0.105 | 1 | 0 | 95 | 1 | 36/65 | 0.550 | 0/2 |
| iter_208 | -3.262 | 1675.65 | 0.14115 | 90665 | 66.00 | 1.200 | 0.050 | 1 | 0 | 40 | 0 | 40/80 | 0.600 | 1/2 |
| iter_175 | -3.463 | 1654.06 | 0.14015 | 85068 | 64.51 | 1.066 | 0.189 | 1 | 0 | 18 | 0 | 26/25 | 0.323 | 0/6 |
| iter_155 | -4.219 | 1648.46 | 0.13950 | 85976 | 58.50 | 0.900 | 0.090 | 1 | 0 | 10 | 0 | 36/73 | 0.395 | 0/1 |
| iter_75 | -5.959 | 1646.24 | 0.13904 | 85449 | 67.85 | 0.860 | 0.095 | 1 | 1 | 39 | 0 | 27/70 | 0.376 | 1/6 |
| iter_212 | -9.642 | 1613.57 | 0.14777 | 83323 | 63.50 | 1.086 | 0.232 | 0 | 0 | 92 | 1 | 40/50 | 0.532 | 0/3 |
| iter_64 | -11.891 | 1632.38 | 0.14252 | 119413 | 51.63 | 1.123 | 0.042 | 0 | 1 | 58 | 0 | 13/46 | 0.596 | 0/3 |
| iter_225 | -12.276 | 1608.10 | 0.14839 | 82865 | 67.60 | 0.810 | 0.068 | 0 | 0 | 92 | 1 | 35/33 | 0.387 | 0/4 |
| iter_106 | -13.022 | 1644.07 | 0.13940 | 84312 | 68.85 | 0.972 | 0.207 | 1 | 0 | 49 | 0 | 35/63 | 0.307 | 1/5 |
| iter_23 | -13.784 | 1619.79 | 0.14155 | 108067 | 57.35 | 1.000 | 0.222 | 0 | 1 | 86 | 0 | 22/28 | 0.494 | 1/5 |
| iter_101 | -14.242 | 1612.01 | 0.
