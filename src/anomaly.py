"""Climate anomaly events, disequilibrium D_t, redistribution lag, speed bias.

D_t primary metric: distance between observed and env-only-predicted spatial
distribution = 1 - Pearson corr(obs_t, envpred_t) over routes (route-demeaned
units). Sensitivity: abundance-weighted centroid distance (km).
Events: route-year env anomaly |t_ann z|>1.5 or drought_z<-1, pooled per
species at year level using cross-species median anomaly years.
"""
import numpy as np, pandas as pd
from core9 import *

EC = ["t_ann", "p_ann", "drought_z", "djf_p", "jun_t"]

def species_frames(p, env, aou):
    d = p[p.AOU == aou][["rid", "Year", "n"]]
    d = d.merge(env, left_on=["rid", "Year"], right_on=["rid", "year"])
    wide = d.pivot_table(index="rid", columns="Year", values="n", aggfunc="sum")
    wide = fill_surveyed_zero(wide)
    E = {c: d.pivot_table(index="rid", columns="Year", values=c) for c in EC}
    return wide, z_anom_base(E), d

def env_pred_frame(y, X, R, T, wide):
    """Forward-chaining predictions -> route x year prediction frame."""
    ye, pred = forward_chain(y, X, R, T, demean=True)
    pf = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for i in np.isfinite(pred).nonzero()[0]:
        pf.loc[R[i], T[i]] = pred[i]
    return pf

def centroid(fr, coords):
    fr = fr.reindex(index=fr.index.intersection(coords.index))
    lat = coords.loc[fr.index, "Latitude"].values
    lon = coords.loc[fr.index, "Longitude"].values
    def cen(col):
        w = fr[col].values
        w = np.where(np.isfinite(w) & (w > 0), w, 0)
        if w.sum() == 0:
            return np.nan, np.nan
        return (w * lat).sum() / w.sum(), (w * lon).sum() / w.sum()
    return pd.DataFrame({yy: cen(yy) for yy in fr.columns},
                        index=["lat", "lon"]).T

def main():
    p = load_parsed()
    env = pd.read_parquet(DATA / "env_all.parquet")
    coords = pd.read_csv(DATA / "route_coords.csv").set_index("rid")
    sp = pd.read_csv(OUT / "species_inclusion.csv")
    inc = sp[sp.included].AOU.astype(str).tolist()
    # prespecified anomaly years: pooled route t_ann anomaly
    za = env.assign(tz=env.groupby("rid").t_ann.transform(
        lambda s: (s - s.mean()) / s.std()))
    yr_z = za.groupby("year").tz.mean()
    events_hot = yr_z[yr_z > 0.7].index.tolist()
    events_cold = yr_z[yr_z < -0.7].index.tolist()
    print("hot years:", events_hot, "cold:", events_cold, flush=True)
    ev_rows, speed_rows, dis_rows = [], [], []
    for aou in inc:
        wide, Ez, d = species_frames(p, env, aou)
        envf = [Ez[c] for c in EC]
        # common evaluation panel: rows with finite env and lag-1 history
        y3, X3, R3, T3 = panel_rows(wide, envf, [1])
        Xenv3 = X3[:, :len(EC)]
        Pf1 = env_pred_frame(y3, Xenv3, R3, T3, wide)
        Pf3 = env_pred_frame(y3, X3, R3, T3, wide)
        occ = occurrence(wide)
        # D_t = 1 - corr(obs share, env-only pred share)
        Dt = {}
        for yy in wide.columns:
            o = occ[yy]; e = Pf1[yy]
            m = o.notna() & e.notna()
            if m.sum() > 50:
                Dt[yy] = 1 - np.corrcoef(o[m], e[m])[0, 1]
        Ds = pd.Series(Dt)
        def sign_flip(fr, ev):
            """fraction of routes whose value crosses its route-level baseline
            (train-only mean over years <= ev) between ev and ev+1."""
            base = fr[[c for c in fr.columns if c <= ev]].mean(axis=1)
            dev = fr[[ev, ev + 1]].sub(base, axis=0)
            return np.sign(dev[ev]), np.sign(dev[ev + 1])
        for ev in events_hot + events_cold:
            if ev not in wide.columns or ev + 1 not in wide.columns:
                continue
            s_o = sign_flip(occ, ev)
            s_e = sign_flip(Pf1, ev)
            s_h = sign_flip(Pf3, ev)
            m = (s_o[0].notna() & s_o[1].notna() & s_e[0].notna()
                 & s_e[1].notna() & s_h[0].notna() & s_h[1].notna())
            if m.sum() < 50:
                continue
            # redistribution = fraction of routes flipping sign vs baseline
            r_obs = (s_o[0][m] != s_o[1][m]).mean()
            r_e = (s_e[0][m] != s_e[1][m]).mean()
            r_h = (s_h[0][m] != s_h[1][m]).mean()
            ev_rows.append({"species": aou, "event_year": ev,
                            "kind": "hot" if ev in events_hot else "cold",
                            "obs_redist": r_obs, "env_pred_redist": r_e,
                            "hist_pred_redist": r_h,
                            "n_routes": int(m.sum())})
            speed_rows.append({"species": aou, "event_year": ev,
                               "speed_bias_env": r_e - r_obs,
                               "speed_bias_hist": r_h - r_obs})
        # disequilibrium decay around event years: mean D before/after
        for ev in events_hot + events_cold:
            for lag in [0, 1, 2, 3]:
                if ev + lag in Ds.index:
                    dis_rows.append({"species": aou, "event_year": ev,
                                     "lag": lag, "D": Ds[ev + lag]})
        print(aou, flush=True)
    pd.DataFrame(ev_rows).to_csv(OUT / "climate_anomaly_redistribution.csv",
                                 index=False)
    pd.DataFrame(speed_rows).to_csv(OUT / "forecast_speed_bias.csv",
                                    index=False)
    pd.DataFrame(dis_rows).to_csv(OUT / "disequilibrium_decay.csv",
                                  index=False)

if __name__ == "__main__":
    main()
