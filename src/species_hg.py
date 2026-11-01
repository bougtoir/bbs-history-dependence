"""2x2 CV x demean diagnostic + species-level HG (primary: FC + route-demean),
lagged-env control, forecast horizons, future-information decomposition,
training-size and window sensitivity.

Outputs:
 cv_demean_2x2_diagnostic.csv   (A,B,C,D per species + effects)
 bbs_species_history_gain.csv   (primary + loyo comparison + CIs)
 future_information_gain.csv    (per-species Delta_future_information)
 forward_training_size_sensitivity.csv
 horizon_hg.csv
"""
import numpy as np, pandas as pd
from core9 import *

EC = ["t_ann", "p_ann", "drought_z", "djf_p", "jun_t"]

def per_year_hg(y, p1, p3, T):
    d = pd.DataFrame({"y": y, "p1": p1, "p3": p3, "t": T})
    d = d[np.isfinite(d.p1) & np.isfinite(d.p3)]
    v = []
    for t, g in d.groupby("t"):
        ss = np.sum(g.y ** 2)
        if ss == 0 or len(g) < 20:
            continue
        r1 = 1 - np.sum((g.y - g.p1) ** 2) / ss
        r3 = 1 - np.sum((g.y - g.p3) ** 2) / ss
        v.append((t, r3 - r1, len(g), np.isfinite(g.y).sum()))
    return pd.DataFrame(v, columns=["year", "hg_t", "n", "nfin"])

def main():
    p = load_parsed()
    env = pd.read_parquet(DATA / "env_all.parquet")
    sp = inclusion_table(p)
    sp.to_csv(OUT / "species_inclusion.csv", index=False)
    inc = sp[sp.included].AOU.tolist()
    print("included:", len(inc), flush=True)
    hg_rows, dec_rows, fut_rows, trn_rows, win_rows, diag_rows = \
        [], [], [], [], [], []
    for aou in inc:
        wide, E = species_matrices(p, env, aou, EC)
        Ez = z_anom_base(E)
        envf = [Ez[c] for c in EC]
        e_lag = [Ez[c].shift(axis=1) for c in EC]
        e_lag2 = e_lag + [Ez[c].shift(2, axis=1) for c in EC]
        # common panel (env + env lags + y lag1)
        yF, XF, RF, TF = panel_rows(wide, envf + e_lag2, [1])
        nE = len(EC)
        Xenv = XF[:, :nE]; Xel1 = XF[:, nE:2 * nE]; Xel2 = XF[:, 2 * nE:3 * nE]
        Xhist = XF[:, -1:]
        E1 = Xenv; E1L1 = np.c_[Xenv, Xel1]; E1L2 = np.c_[Xenv, Xel1, Xel2]
        # --- 2x2 diagnostic: same rows/predictors
        def r2_fc(Xc, demean, **kw):
            ye, pr = forward_chain(yF, Xc, RF, TF, demean=demean, **kw)
            return r2_from_pred(ye, pr)
        def r2_ly(Xc, demean, **kw):
            ye, pr = loyo_pred(yF, Xc, RF, TF, demean=demean, **kw)
            return r2_from_pred(ye, pr)
        XE, XH = Xenv, np.c_[Xenv, Xhist]
        A = r2_ly(XH, False) - r2_ly(XE, False)
        B = r2_ly(XH, True) - r2_ly(XE, True)
        C = r2_fc(XH, False) - r2_fc(XE, False)
        D = r2_fc(XH, True) - r2_fc(XE, True)
        A8 = r2_ly(XH, False, year_demean=True) - \
            r2_ly(XE, False, year_demean=True)  # Phase-8B style
        diag_rows.append({"species": aou, "HG_A": A, "HG_B": B, "HG_C": C,
                          "HG_D": D, "HG_A_8Bstyle": A8,
                          "n_years": wide.shape[1], "n_routes": wide.shape[0]})
        # --- main table (D = primary) + lagged-env incremental
        m1 = r2_fc(Xenv, True); m3 = r2_fc(XH, True)
        m5 = r2_fc(E1L1, True); m6 = r2_fc(np.c_[E1L1, Xhist], True)
        m6b = r2_fc(np.c_[E1L2, Xhist], True)
        m5b = r2_fc(E1L2, True)
        m0 = r2_fc(np.zeros((len(yF), 0)), True)
        m2 = r2_fc(Xhist, True)
        m1nd = r2_fc(Xenv, False); m3nd = r2_fc(XH, False)
        ye1, p1 = forward_chain(yF, Xenv, RF, TF, demean=True)
        ye3, p3 = forward_chain(yF, XH, RF, TF, demean=True)
        py = per_year_hg(ye3, p1, p3, TF)
        v = py.hg_t.values if len(py) else np.array([np.nan])
        hg = np.nanmean(v)
        lo, hi = (np.nanpercentile(v, [2.5, 97.5])
                  if len(v) >= 5 else (np.nan, np.nan))
        se = np.nanstd(v, ddof=1) / np.sqrt(len(v)) if len(v) >= 5 else np.nan
        hg_rows.append({"species": aou, "n_routes": wide.shape[0],
                        "n_years": wide.shape[1],
                        "M0": m0, "M1": m1, "M2": m2, "M3": m3,
                        "M5": m5, "M5b": m5b, "M6": m6, "M6b": m6b,
                        "HG": m3 - m1, "HG_lo": hg - 1.96 * se,
                        "HG_hi": hg + 1.96 * se,
                        "EG": m3 - m2,
                        "HG_lagenv": m6 - m5, "HG_lagenv2": m6b - m5b,
                        "M1_nodemean": m1nd, "M3_nodemean": m3nd})
        # --- future-information: per target year, past-only vs past+future
        yl1, pl1 = loyo_pred(yF, Xenv, RF, TF, demean=True)
        yl3, pl3 = loyo_pred(yF, XH, RF, TF, demean=True)
        pyF = per_year_hg(ye3, p1, p3, TF).set_index("year").hg_t
        pyL = per_year_hg(yl3, pl1, pl3, TF).set_index("year").hg_t
        com = pyF.index.intersection(pyL.index)
        if len(com) >= 5:
            fut_rows.append({"species": aou,
                             "HG_past_only": pyF[com].mean(),
                             "HG_past_future": pyL[com].mean(),
                             "Delta_future_information":
                             pyL[com].mean() - pyF[com].mean()})
        # --- horizons (FC, demean)
        for h in [1, 2, 3, 5]:
            # common evaluation panel for this horizon
            yh, Xh, Rh, Th = panel_rows(wide, envf, [h])
            Xhe = Xh[:, :nE]
            ye_, pe_ = forward_chain(yh, Xh, Rh, Th, demean=True)
            ye2, pe2 = forward_chain(yh, Xhe, Rh, Th, demean=True)
            dec_rows.append({"species": aou, "horizon": h,
                             "HG_h": r2_from_pred(ye_, pe_)
                             - r2_from_pred(ye2, pe2)})
        # --- training-size sensitivity: HG by min_hist
        for mh in [5, 10, 15]:
            ye_, p_ = forward_chain(yF, Xenv, RF, TF, demean=True,
                                    min_hist=mh)
            ye3_, p3_ = forward_chain(yF, XH, RF, TF, demean=True,
                                      min_hist=mh)
            trn_rows.append({"species": aou, "min_hist": mh,
                             "HG": r2_from_pred(ye3_, p3_)
                             - r2_from_pred(ye_, p_)})
        # per-target-year diagnostics
        for _, rr in py.iterrows():
            trn_rows.append({"species": aou, "min_hist": np.nan,
                             "target_year": int(rr.year),
                             "train_years": int(rr.year - TF.min()),
                             "hg_t": rr.hg_t})
        # --- rolling vs expanding
        for w in [None, 10, 20]:
            ye_, p_ = forward_chain(yF, Xenv, RF, TF, demean=True, window=w)
            ye3_, p3_ = forward_chain(yF, XH, RF, TF, demean=True, window=w)
            win_rows.append({"species": aou, "window": w or "expand",
                             "HG": r2_from_pred(ye3_, p3_)
                             - r2_from_pred(ye_, p_)})
        print(aou, f"D={D:.3f} A={A:.3f}", flush=True)
    pd.DataFrame(diag_rows).to_csv(OUT / "cv_demean_2x2_diagnostic.csv",
                                   index=False)
    pd.DataFrame(hg_rows).to_csv(OUT / "bbs_species_history_gain.csv",
                                 index=False)
    pd.DataFrame(fut_rows).to_csv(OUT / "future_information_gain.csv",
                                  index=False)
    pd.DataFrame(trn_rows).to_csv(OUT / "forward_training_size_sensitivity.csv",
                                  index=False)
    pd.DataFrame(dec_rows).to_csv(OUT / "horizon_hg.csv", index=False)
    pd.DataFrame(win_rows).to_csv(OUT / "window_sensitivity.csv", index=False)
    d = pd.DataFrame(diag_rows)
    print(d[["HG_A", "HG_B", "HG_C", "HG_D", "HG_A_8Bstyle"]].describe())

if __name__ == "__main__":
    main()
