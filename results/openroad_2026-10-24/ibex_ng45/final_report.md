# Final DSE Report

## Executive Summary

The run is operationally clean: **230/230 jobs succeeded, 230/230 are usable, and 0 are unusable**. However, only **76/230 are full_valid**; the other **154 are quick/CTS-partial** points and should not be used for final Pareto claims. The full-valid Pareto set contains **15 points**, all of which still miss timing.

The raw reported best WNS is **iter_162 at +0.01158**, but it is a **quick partial** point with **ABC_AREA=1** and an anomalous power of **0.06986 W**, roughly 3.5x the normal ~0.020–0.021 W cluster. It is therefore not a valid closure result. The best **full_valid Pareto WNS** is **iter_123 at -0.01075**, with area **29030.2** and power **0.02107 W**.

The full Pareto frontier is timing-limited: every full-valid Pareto point has negative WNS, ranging from **-0.01075 to -0.03848**. Area is relatively flat at **28645–29153**, and power is also flat at **0.02036–0.02147 W**. The search has found a stable feasible region but is plateauing: cumulative hypervolume gains become tiny after roughly iter_158, and many promoted points regress versus their quick parent.

## Search Quality and Data Quality

**Search quality**

- **Operational success is excellent:** 230/230 job_success, 0 job_fail, 230 usable.
- **Full-valid coverage is moderate:** 76 full-valid points out of 230 launched, about 33%.
- **Pareto yield is reasonable but not large:** 15 Pareto points among 76 full-valid points, about 20%.
- **The search is saturating:** cumulative hypervolume rises from about **247.214M** early to **247.289M** late; the largest visible jump is around iter_123, after which gains are mostly small. Several later Pareto additions have very small EHVI deltas.
- **Many promotions regress versus quick parents:** `qor_regressed=true` is common. This is not a job failure; it is quick-to-full fidelity mismatch and multi-objective regression.
- **No full-valid point closes timing.** The “best WNS” full point is still negative, so the real search outcome is “least-negative WNS,” not timing closure.

**Data quality**

- **No unusable points and no failure snippets:** `fail_log_tail` is empty, `fail_stage` is empty, and no job failures are present.
- **Quick data is partial:** 154 quick points use `4_1_cts.json`, have `metrics_status=partial`, `metrics_valid=false`, and missing `wl` (`Infinity`). They are useful for triage but not for final Pareto comparison.
- **Full data is complete enough for Pareto analysis:** 76 full points use `6_report.json`, have finite `wl`, and are full_valid.
- **Clock-period fields are broken globally:** `clock_period=Infinity` and `ecp=Infinity` for all points. WNS/area/power are still usable, but derived timing margin is not.
- **ABC_AREA=1 creates a power outlier:** normal full points are near 0.020–0.021 W; ABC_AREA=1 points such as iter_162 and iter_185 are near 0.070 W. Until this is explained, ABC_AREA=1 should be quarantined for power comparisons.
- **Log symptoms are nonfatal but persistent:** most points report `overflow`, `congestion`, and `setup` symptoms; quick points have proxy route overflow around 390–440. These did not produce job failures, but they indicate the search is operating near routing/placement pressure.

## Pareto Frontier Insights

The full Pareto frontier is dominated by small timing/area/power trade-offs rather than a large area-power spread.

| Representative point | Role | WNS | Area | Power |
|---|---|---:|---:|---:|
| iter_123 | Best full-valid WNS | -0.01075 | 29030.2 | 0.02107 |
| iter_150 | Pareto area-min | -0.03848 | 28645.0 | 0.02038 |
| iter_158 | Pareto power-min | -0.03253 | 28667.9 | 0.02036 |

Interpretation:

- Improving WNS from the area-min point iter_150 to the best-WNS point iter_123 costs about **+385 area (+1.3%)** and **+0.00069 W (+3.4%)**, while improving WNS by about **0.0277**.
- Moving from the power-min point iter_158 to iter_123 costs about **+362 area (+1.3%)** and **+0.00071 W (+3.5%)**, while improving WNS by about **0.0218**.
- The area-min and power-min points are very close: iter_150 saves only about **22.9 area** versus iter_158 but has about **0.0060 worse WNS** and marginally higher power. This is a weak area trade-off.
- Because every full Pareto point has negative WNS, the practical trade-off is “how much area/power to spend to reduce the timing deficit,” not “closed timing with area/power alternatives.”

## Parameter Sensitivity

| Parameter | Apparent effect in this run | Evidence and caution |
|---|---|---|
| CORE_UTILIZATION | Higher utilization favors area; lower utilization favors WNS. | Best full WNS iter_123 uses **44.7**; area/power-favored points cluster at **58–60** but have WNS around -0.029 to -0.038. |
| CORE_ASPECT_RATIO | High aspect ratio appears in area-favored points; best WNS uses ~1.195. | Best WNS iter_123 AR=1.195; low-area iter_150 AR=1.186; but some good-WNS points use AR≈0.84–1.09, so interaction with utilization matters. |
| CORE_MARGIN | Weak/no clean monotonic effect. | Best WNS uses margin 8; area/power-favored points often use 3–5. Confounded with utilization and aspect ratio. |
| PLACE_DENSITY_LB_ADDON | Moderate 0.20–0.25 dominates Pareto; very low values hurt. | Best WNS uses 0.241; low-area/power points use 0.208–0.248. Very low 0.06 gives worse WNS in iter_72; 0.1 appears in high-power iter_171. |
| place_padding | Padding=1 tends to increase power and often does not improve WNS. | Best WNS iter_123 and low-area/power iter_150/iter_158/iter_132 use padding=0. Several padding=1 points have power near 0.0210–0.0215. |
| ROUTING_LAYER_ADJUSTMENT | Not monotonic in the 15-point full Pareto. | Best WNS uses 0.385; many Pareto points use 0.55–0.60, but those often have higher power. Low values 0.32–0.42 appear in best WNS and low-area/power points. |
| cts_cluster_size / CTS_CLUSTER_DIAMETER | Small CTS diameter appears in best WNS but is not sufficient alone. | Best WNS uses size 25/diameter 55. However, severe setup outlier iter_211 also uses diameter 55, so padding/density/utilization interactions dominate. |
| TNS_END_PERCENT | No clear monotonic benefit from very high values. | Best WNS uses TNS=35; low-area/power points use 70–89; TNS=100 appears in non-frontier or worse-WNS points such as iter_72. |
| ABC_AREA | Strongly associated with abnormal reported power. | All 15 full Pareto points use ABC_AREA=0. ABC_AREA=1 points report ~0.070 W and should be excluded from power comparisons until validated. |
| gp_timing_driven | Fixed at 1 for all Pareto points. | No variance in the Pareto set; keep enabled, but this data cannot estimate its isolated effect. |
| gp_routability_driven | No clear benefit in this sample. | Only iter_102 in the Pareto set uses 1; most Pareto points use 0. Do not enable globally without evidence. |

Overall, the strongest actionable controls appear to be **CORE_UTILIZATION**, **place_padding**, **PLACE_DENSITY_LB_ADDON**, and **ABC_AREA quarantine**. RLA and CTS effects are present but confounded by utilization, padding, and density.

## Failure Analysis

There are **no fatal job failures** in this dataset: 230/230 succeeded, 0 failed, and no `fail_log_tail` snippets exist. The “failure” modes are therefore optimization and fidelity failures, not tool crashes.

- **Dominant failure mode: quick-to-full WNS regression.** Quick CTS WNS is often optimistic versus full finish WNS.
  - iter_140 quick WNS **-0.00199**; its full child iter_150 reached **-0.03848**, a WNS change of about **-0.03649**, with area **+41.8** and power **+0.00042**.
  - iter_179 quick WNS **-0.00876**; full child iter_190 reached **-0.02647**, a WNS change of about **-0.01771**.
  - iter_208 quick WNS **-0.04217**; full child iter_211 reached **-0.08134**, a WNS change of about **-0.03917**, with area **+216.5**.
- **Some full points improve WNS versus parent but still regress overall.** iter_210 shows `qor_vs_parent` WNS **+0.01066**, but area **+86.7** and power **+0.000548**, and is still flagged `qor_regressed=true`. This is multi-objective regression, not a WNS failure.
- **ABC_AREA=1 is a data/failure quarantine case.** iter_162 has attractive quick WNS **+0.01158**, but power **0.06986 W** and partial status. iter_185 is full_valid but reports power **0.07067 W** with WNS **-0.01959**. Until the power metric is understood, ABC_AREA=1 should not drive power or Pareto decisions.
- **Severe setup outliers exist outside the Pareto set.** Examples include iter_211 WNS **-0.08134**, iter_182 **-0.06198**, and iter_210 **-0.05776**. These often combine padding=1, low density add-on, or large CTS diameter, but the sample is too small to isolate one cause.
- **Overflow/congestion symptoms are persistent but nonfatal.** Most `log_summary.symptoms` include `overflow`, `congestion`, and `setup`; quick proxy route overflow is around 390–440. Since there are no route/DRC job failures, this is best treated as a placement/routing pressure warning, not a hard failure.

## Recommended Next Experiments

1. **Promote the strongest normal-power quick candidates to full finish before trusting them.** Prioritize iter_220 (quick WNS +0.00145, normal power, ABC_AREA=0), iter_202, iter_203, iter_175, iter_169, iter_153, and iter_194. These are more credible closure candidates than iter_162 because they avoid the ABC_AREA=1 power anomaly.
2. **Quarantine ABC_AREA=1 from Pareto/power decisions.** Rerun iter_162 and similar ABC_AREA=1 points with ABC_AREA=0, or fix/validate the power extraction path. Do not compare their ~0.070 W against normal ~0.020 W points.
3. **Intensify around the best full-valid WNS seed iter_123.** Test local variants around CORE_UTILIZATION 44–50, CORE_ASPECT_RATIO 1.15–1.20, CORE_MARGIN 7–8, PLACE_DENSITY_LB_ADDON 0.23–0.25, place_padding=0, RLA 0.35–0.45, and small CTS clusters.
4. **Run a focused area/power branch from iter_158 and iter_132.** Keep high utilization 58–60, aspect ratio 1.16–1.19, place_padding=0, density 0.21–0.25, and RLA 0.33–0.40. Treat these as area/power seeds but expect WNS around -0.03 unless timing recovery is added.
5. **Do a controlled A/B on place_padding.** The data suggests padding=1 raises power and rarely improves WNS. Test paired runs with identical parameters except padding 0 vs 1, then keep padding only if it buys measurable WNS or routability.
6. **Retire very low PLACE_DENSITY_LB_ADDON and very high TNS_END_PERCENT combinations unless they fix a specific violation.** Low density add-on around 0.02–0.10 and TNS=90–100 are not producing better full Pareto points in this sample.
7. **Fix metric extraction before the next campaign.** `clock_period` and `ecp` are Infinity for all points, and quick `wl` is Infinity. Extract the actual Nangate45 SDC period (2.2 ns or the active scenario period) and compute real margin/ECP.
8. **Use full-valid-only correlation/DOE analysis.** Recompute parameter sensitivities on the 76 full_valid points only, with ABC_AREA=1 excluded, and model interactions among utilization, aspect ratio, padding, density, and RLA.
9. **Treat overflow/congestion as a soft constraint, not a global objective.** Since there are no route failures, avoid enabling `gp_routability_driven` globally. Instead test it only on points with high proxy overflow and verify it does not sacrifice WNS/area/power.
10. **Shift from broad exploration to local intensification and confirmation.** Hypervolume gains are saturating. A smaller batch of high-value full runs around iter_123, iter_158, iter_132, and the best normal-power quick candidates will likely be more productive than continued wide sampling.

## Appendix: Top Pareto Points

All rows below are full_valid and Pareto. Sorted by WNS from best to worst. The raw quick best WNS iter_162 is intentionally excluded because it is partial and not full_valid.

| Run | WNS | Area | Power | CU | AR | Margin | Density addon | Pad | RLA | CTS size/diam | TNS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iter_123 | -0.01075 | 29030.2 | 0.02107 | 44.70 | 1.195 | 8 | 0.241 | 0 | 0.385 | 25/55 | 35 |
| iter_213 | -0.01423 | 28954.1 | 0.02122 | 58.20 | 0.867 | 6 | 0.2346 | 1 | 0.5528 | 26/116 | 90 |
| iter_171 | -0.01654 | 28882.5 | 0.02147 | 58.50 | 0.840 | 3 | 0.100 | 1 | 0.600 | 22/95 | 75 |
| iter_71 | -0.01661 | 29153.3 | 0.02105 | 55.75 | 0.875 | 8 | 0.1938 | 1 | 0.5998 | 18/148 | 41 |
| iter_218 | -0.01783 | 28974.8 | 0.02102 | 55.00 | 1.090 | 5 | 0.245 | 1 | 0.320 | 19/114 | 90 |
| iter_135 | -0.01873 | 28923.0 | 0.02077 | 53.50 | 0.950 | 7 | 0.200 | 0 | 0.600 | 37/120 | 55 |
| iter_29 | -0.02143 | 28810.5 | 0.02114 | 53.65 | 0.830 | 6 | 0.2015 | 1 | 0.5781 | 31/134 | 82 |
| iter_43 | -0.02302 | 28910.2 | 0.02098 | 58.56 | 1.014 | 7 | 0.2113 | 1 | 0.5474 | 18/144 | 50 |
| iter_102 | -0.02310 | 28874.3 | 0.02091 | 56.63 | 1.160 | 6 | 0.2437 | 1 | 0.3616 | 33/106 | 79 |
| iter_72 | -0.02577 | 28939.5 | 0.02074 | 58.00 | 1.000 | 4 | 0.060 | 0 | 0.500 | 20/100 | 100 |
| iter_190 | -0.02647 | 28782.8 | 0.02074 | 49.65 | 1.012 | 5 | 0.2283 | 0 | 0.4206 | 23/101 | 80 |
| iter_107 | -0.02839 | 28754.6 | 0.02059 | 53.20 | 1.050 | 5 | 0.2484 | 0 | 0.4128 | 30/138 | 14 |
| iter_132 | -0.02912 | 28755.9 | 0.02039 | 54.89 | 1.177 | 3 | 0.208 | 0 | 0.329 | 32/137 | 70 |
| iter_158 | -0.03253 | 28667.9 | 0.02036 | 59.00 | 1.162 | 5 | 0.2422 | 0 | 0.3669 | 26/126 | 89 |
| iter_150 | -0.03848 | 28645.0 | 0.02038 | 59.76 | 1.186 | 5 | 0.2259 | 0 | 0.3996 | 39/125 | 79 |
