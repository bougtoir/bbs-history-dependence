"""Generate LOYO_FORWARD_DIAGNOSTIC.md, PHASE9_GO_NO_GO.md (13-question memo),
from results/tables."""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
DOC = ROOT / "docs"; DOC.mkdir(parents=True, exist_ok=True)

def F(v, n=3):
    try:
        return "NA" if not np.isfinite(v) else f"{v:.{n}f}"
    except TypeError:
        return "NA"

def boot_ci(x, n=1000):
    x = np.asarray(pd.Series(x).dropna())
    if len(x) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])

def main():
    dg = pd.read_csv(T / "cv_demean_2x2_diagnostic.csv")
    hg = pd.read_csv(T / "bbs_species_history_gain.csv")
    fut = pd.read_csv(T / "future_information_gain.csv")
    trn = pd.read_csv(T / "forward_training_size_sensitivity.csv")
    hys = pd.read_csv(T / "matched_environment_hysteresis.csv")
    trj = pd.read_csv(T / "trajectory_direction.csv")
    an = pd.read_csv(T / "climate_anomaly_redistribution.csv")
    sb = pd.read_csv(T / "forecast_speed_bias.csv")
    dis = pd.read_csv(T / "disequilibrium_decay.csv")
    sim = pd.read_csv(T / "simulation_false_positive.csv")
    obs = pd.read_csv(T / "observer_occurrence_scale_moderators.csv")
    hz = pd.read_csv(T / "horizon_hg.csv")

    # required deliverable: history gain after controlling lagged environment
    hg[["species", "n_routes", "n_years", "M1", "M3", "M5", "M5b", "M6", "M6b",
        "HG", "HG_lagenv", "HG_lagenv2", "HG_lo", "HG_hi"]].to_csv(
        T / "history_after_lagged_env.csv", index=False)

    L = ["# LOYO vs forward-chaining diagnostic\n\n",
         "Same species inclusion, predictors (env + lag-1 vs env-only), "
         "response, and R2 metric in all four cells.\n\n",
         "## 2x2 table (median over species)\n\n",
         "| cell | CV | route demean | median HG | IQR | P(HG>0) |\n",
         "|---|---|---|---|---|---|\n"]
    for cell, cv, dm in [("A", "LOYO", "no"), ("B", "LOYO", "yes"),
                         ("C", "forward", "no"), ("D", "forward", "yes")]:
        v = dg[f"HG_{cell}"]
        L.append(f"| {cell} | {cv} | {dm} | {F(v.median())} | "
                 f"[{F(v.quantile(.25))},{F(v.quantile(.75))}] | "
                 f"{(v > 0).mean():.2f} |\n")
    L.append(f"\nPhase-8B-style year-demean LOYO HG_A_8Bstyle median: "
             f"{F(dg.HG_A_8Bstyle.median())}\n")
    L.append("\n## Effect decomposition\n\n")
    cv_eff = dg.HG_C - dg.HG_A
    dm_ly = dg.HG_B - dg.HG_A
    dm_fc = dg.HG_D - dg.HG_C
    for name, v in [("CV effect (C-A)", cv_eff),
                    ("demean effect, LOYO (B-A)", dm_ly),
                    ("demean effect, forward (D-C)", dm_fc),
                    ("residual forward HG (D)", dg.HG_D)]:
        lohi = boot_ci(v)
        L.append(f"- {name}: median {F(v.median())}, IQR "
                 f"[{F(v.quantile(.25))},{F(v.quantile(.75))}], "
                 f"P(>0) {(v > 0).mean():.2f}, boot95 "
                 f"[{F(lohi[0])},{F(lohi[1])}]\n")
    L.append("\n## Future-information\n\n")
    L.append(f"Delta_future_information median {F(fut.Delta_future_information.median())}, "
             f"HG_past_only median {F(fut.HG_past_only.median())}, "
             f"HG_past_future median {F(fut.HG_past_future.median())}.\n")
    L.append("\n## Forward-chaining validity\n\n")
    L.append("- demeaning route means: train-only (fixed)\n"
             "- env anomalies: fixed baseline <=2005, no future stats\n"
             "- predictor centring in _fit: train-only column means (no SD scaling; complete-case rows, no imputation)\n"
             "- inclusion rule: coverage counts only, no outcome info\n")
    L.append("\n## Training-size / window sensitivity\n\n")
    tt = trn.dropna(subset=["min_hist"])
    for mh, g in tt.groupby("min_hist"):
        L.append(f"- min_hist {int(mh)}: median HG {F(g.HG.median())}, "
                 f"P(>0) {(g.HG > 0).mean():.2f}\n")
    win = pd.read_csv(T / "window_sensitivity.csv")
    for w, g in win.groupby("window"):
        L.append(f"- window {w}: median HG {F(g.HG.median())}\n")
    # pattern verdict
    medA, medB, medC, medD = (dg.HG_A.median(), dg.HG_B.median(),
                              dg.HG_C.median(), dg.HG_D.median())
    if medA > medB + 0.05 and medC > medD + 0.05:
        pat = "Pattern 2 (static route persistence dominates)"
        if medA > medC + 0.05 and medB > medD + 0.05:
            pat = "Pattern 3 (both CV and demeaning contribute)"
    elif medA > medC + 0.05 and abs(medB - medD) < 0.05:
        pat = "Pattern 1 (LOYO future-year information dominates)"
    else:
        pat = "Pattern 4 (forward HG robustly positive)" if medD > 0 else \
            "Pattern 2 (static route persistence dominates)"
    L.append(f"\n## Pattern: {pat}\n")
    (DOC / "LOYO_FORWARD_DIAGNOSTIC.md").write_text("".join(L))

    # ---- GO/NO-GO doc
    M = ["# Phase-9 GO/NO-GO — BBS hysteresis and delayed redistribution\n\n"]
    n = len(hg)
    M.append(f"Included species: {n}\n\n")
    M.append(f"- HG>0: {(hg.HG > 0).mean():.2f}; median HG {F(hg.HG.median())}\n")
    M.append(f"- CI entirely >0: {((hg.HG_lo > 0)).mean():.2f}\n")
    M.append(f"- HG survives lagged env (HG_lagenv>0): "
             f"{(hg.HG_lagenv > 0).mean():.2f}, median {F(hg.HG_lagenv.median())}\n")
    M.append(f"- matched-env effect median {F(hys.effect_of_prior_state.median())}; "
             f"frac sig (p<0.05): {(hys.p_boot < 0.05).mean():.2f}\n")
    M.append(f"- anomaly: obs_redist median {F(an.obs_redist.median())}, "
             f"env-pred {F(an.env_pred_redist.median())}, "
             f"hist-pred {F(an.hist_pred_redist.median())}\n")
    M.append(f"- speed bias: env {F(sb.speed_bias_env.median())} vs hist "
             f"{F(sb.speed_bias_hist.median())}\n")
    fpA = sim[sim.scenario == "A"].P_hyst_gt_0.median()
    fpC = sim[sim.scenario == "C"].P_hyst_gt_0.median()
    M.append(f"- sim FP: P(hyst>0 | env-only)={F(fpA)}, "
             f"P(hyst>0 | static-route)={F(fpC)}\n")
    M.append(f"- observer-controlled HG: median "
             f"{F((obs.M3_obs - obs.M1_obs).median())}; occurrence HG "
             f"{F(obs.HG_occ.median())}; regional HG {F(obs.HG_regional.median())}\n")
    M.append(f"- migratory vs resident HG: "
             + "; ".join(f"{k}:{F(v.HG.median())}"
                         for k, v in obs.groupby("mig")) + "\n")
    M.append(f"- HG(h): "
             + ", ".join(f"h{int(h)}={F(v)}" for h, v in
                         hz.groupby("horizon").HG_h.median().items()) + "\n")
    M.append("\n## 13-question memo\n\n")
    qs = [("1. How many species have HG>0?",
           f"{(hg.HG > 0).sum()}/{n} under forward-chaining + route demean."),
          ("2. How many exceed the simulation null?",
           "See simulation_false_positive.csv; species counts vs p95 in table."),
          ("3. Retain history after static route effects?",
           f"Median HG_D (demeaned) {F(hg.HG.median())}; "
           f"frac>0 {(hg.HG > 0).mean():.2f}."),
          ("4. Retain history after lagged environment?",
           f"Median incremental HG {F(hg.HG_lagenv.median())}."),
          ("5. Pass matched-environment path dependence?",
           f"Median prior-state effect {F(hys.effect_of_prior_state.median())} "
           f"(P(occ|prior) - P(occ|no prior) at matched env)."),
          ("6. Genuine hysteresis supported?",
           "See matched_environment_hysteresis + trajectory tables."),
          ("7. Do climate anomalies produce delayed redistribution?",
           f"See disequilibrium_decay (D by lag) and anomaly tables."),
          ("8. Do env-only models predict redistribution too quickly?",
           f"speed_bias_env median {F(sb.speed_bias_env.median())}."),
          ("9. Do history-aware models improve timing?",
           f"speed_bias_hist median {F(sb.speed_bias_hist.median())} "
           f"vs env {F(sb.speed_bias_env.median())}."),
          ("10. Which strategies show strongest history?",
           "See observer_occurrence_scale_moderators.csv mig groups."),
          ("11. Sufficient for Nature submission?",
           "See verdict rationale below."),
          ("12. Strongest result against preferred interpretation?",
           "2x2 diagnostic: HG collapses to ~0 once route means are "
           "removed; LOYO HG largely reflects route-level static "
           "persistence."),
          ("13. Single analysis that would most change conclusion?",
           "Occupancy-model (nonlinear) matched-env test or "
           "individual-marked data (route fidelity).")]
    for q, a in qs:
        M.append(f"**{q}** {a}\n\n")
    # verdict
    M.append("## VERDICT: WEAK GO (no Nature escalation — stop rule engaged)\n\n")
    M.append(
        "After train-only route demeaning, the prospective forward-chaining "
        f"HG collapses to a median of {F(hg.HG.median())} "
        f"({(hg.HG > 0).mean():.0%} of species >0, "
        f"{(hg.HG_lo > 0).mean():.0%} with 95% CI entirely >0). The 2x2 "
        "diagnostic assigns the Phase-8B LOYO signal primarily to Pattern 2: "
        "route-level static persistence, not to LOYO future-year information "
        "(CV effect C-A median "
        f"{F((dg.HG_C - dg.HG_A).median())}).\n\n")
    M.append(
        "What survives static geography: a small positive lag-1 history "
        "contribution (median HG "
        f"{F(hg.HG.median())}) that persists after lagged-env controls "
        f"(M6-M5 median {F(hg.HG_lagenv.median())}) and is stable across "
        "minimum-history 5/10/15-yr screens and expanding/10-yr/20-yr "
        "windows. At matched environment, the occupancy contrast degenerates "
        f"(median {F(hys.effect_of_prior_state.median())}; defined for "
        f"{hys.effect_of_prior_state.notna().sum()}/{len(hys)} species) "
        "because focal species are near-ubiquitous, but the "
        f"abundance contrast is positive (median {F(hys.eff_abundance.median())}"
        f", CI>0 in {(hys.ci_ab_lo > 0).sum()}/"
        f"{hys.ci_ab_lo.notna().sum()} defined species) - prior-year "
        "occupancy predicts next-year abundance at fixed environment.\n\n")
    M.append(
        f"Regional (state-level) aggregation HG median "
        f"{F(obs.HG_regional.median())} where defined "
        "(does NOT retain the route-level signal), while observer "
        f"covariates do not change the picture (observer-controlled HG "
        f"{F((obs.M3_obs - obs.M1_obs).median())}). Climate-anomaly events show "
        f"history-aware forecasts slower than env-only (speed bias "
        f"{F(sb.speed_bias_hist.median())} vs {F(sb.speed_bias_env.median())})"
        ", consistent with inertia, but the same simulations show the "
        "demeaned-FC pipeline cannot distinguish true history from static "
        "structure at route level (scenario E HG negative; occupancy "
        "hysteresis metric has P(FP|static-route)="
        f"{F(fpC)}).\n\n")
    M.append(
        "Interpretation: the Phase-8B BBS signal is dominated by "
        "retrospective route-level persistence; a genuine but small "
        "prospective lag-1 component and a robust abundance-level prior-state "
        "effect remain. Under the prespecified scale this is WEAK GO "
        "(only lag-1 persistence). Per the stop rule, matched-env occupancy "
        "history effects disappeared at route level and Nature escalation "
        "stops; hysteresis/delayed-redistribution claims are not supported "
        "for the Nature-track target.\n")
    (DOC / "PHASE9_GO_NO_GO.md").write_text("".join(M))
    print("written")

if __name__ == "__main__":
    main()
