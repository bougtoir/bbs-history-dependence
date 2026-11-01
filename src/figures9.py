"""Phase-9 figures 1-9."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core9 import OUT, ROOT

FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
T = ROOT / "results" / "tables"

def sv(fig, name):
    fig.savefig(FIG / name, dpi=140, bbox_inches="tight"); plt.close(fig)

def main():
    hg = pd.read_csv(T / "bbs_species_history_gain.csv")
    hyst = pd.read_csv(T / "matched_environment_hysteresis.csv")
    sim = pd.read_csv(T / "simulation_false_positive.csv")
    dec = pd.read_csv(ROOT / "results" / "tables" / "horizon_hg.csv") if (T / "horizon_hg.csv").exists() else None
    an = pd.read_csv(T / "climate_anomaly_redistribution.csv")
    sb = pd.read_csv(T / "forecast_speed_bias.csv")
    obs = pd.read_csv(T / "observer_occurrence_scale_moderators.csv")

    f, ax = plt.subplots()
    ax.hist(hg.HG.dropna(), bins=30); ax.axvline(0, color="k")
    ax.set_xlabel("HG (forward-chaining)"); ax.set_title("Fig1: species-level HG")
    sv(f, "fig1_hg_dist.png")

    f, ax = plt.subplots()
    ax.errorbar(range(len(hg)), hg.sort_values("HG").HG.reset_index(drop=True),
                fmt=".")
    ax.axhline(0, color="r"); ax.set_title("Fig2: HG + lagged-env adjusted")
    ax.plot(range(len(hg)), hg.sort_values("HG").HG_lagenv.reset_index(drop=True),
            "r.", alpha=0.5); sv(f, "fig2_adjusted.png")

    f, ax = plt.subplots()
    ax.hist(hyst.effect_of_prior_state.dropna(), bins=30)
    ax.axvline(0, color="k")
    ax.set_title("Fig3: matched-env prior-state effect"); sv(f, "fig3_matched.png")

    tr = pd.read_csv(T / "trajectory_direction.csv")
    f, ax = plt.subplots()
    ax.scatter(tr.d_occ_warming, tr.d_occ_cooling, alpha=0.4)
    lim = np.nanmax(abs(pd.concat([tr.d_occ_warming, tr.d_occ_cooling])))
    ax.plot([-lim, lim], [-lim, lim], "k--")
    ax.set_xlabel("d_occ warming"); ax.set_ylabel("d_occ cooling")
    ax.set_title("Fig4: trajectory direction"); sv(f, "fig4_traj.png")

    f, ax = plt.subplots()
    for k, g in an.groupby("kind"):
        ax.hist(g.obs_redist, bins=20, alpha=0.4, label=k)
    ax.legend(); ax.set_title("Fig5: anomaly redistribution"); sv(f, "fig5_events.png")

    f, ax = plt.subplots()
    m = an.dropna()
    ax.scatter(m.env_pred_redist, m.obs_redist, alpha=0.2, label="env-only")
    ax.scatter(m.hist_pred_redist, m.obs_redist, alpha=0.2, label="env+hist")
    ax.plot([0, m.obs_redist.max()], [0, m.obs_redist.max()], "k--")
    ax.legend(); ax.set_title("Fig6: predicted vs observed redistribution")
    sv(f, "fig6_redist.png")

    if dec is not None:
        f, ax = plt.subplots()
        dec.groupby("horizon").HG_h.median().plot(ax=ax, marker="o")
        ax.set_title("Fig7: HG by horizon"); sv(f, "fig7_horizon.png")

    f, ax = plt.subplots()
    dd = pd.read_csv(T / "disequilibrium_decay.csv")
    dd.groupby("lag").D.median().plot(ax=ax, marker="o")
    ax.set_title("Fig8: disequilibrium decay"); sv(f, "fig8_lag.png")

    f, ax = plt.subplots()
    piv = sim.groupby("scenario").median_HG.median()
    piv.plot.bar(ax=ax); ax.set_title("Fig9: sim HG by scenario")
    sv(f, "fig9_sim.png")

if __name__ == "__main__":
    main()
