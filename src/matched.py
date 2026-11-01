"""Matched-environment hysteresis + ascending/descending trajectory tests."""
import numpy as np, pandas as pd
from core9 import *

RNG = np.random.default_rng(1)

def match_effect(wide, Ez, envcols, tol=0.5):
    """Within-route: years where |env z| <= tol on all vars; compare
    occurrence/abundance by prior-year occupancy."""
    occ = occurrence(wide)
    rows = []
    for rid in wide.index:
        for yy in wide.columns:
            if yy - 1 not in wide.columns:
                continue
            good = all((yy in Ez[c].columns) and
                       np.isfinite(Ez[c].loc[rid, yy]) and
                       abs(Ez[c].loc[rid, yy]) <= tol
                       for c in envcols)
            if not good:
                continue
            o_t, o_tm1 = occ.loc[rid, yy], occ.loc[rid, yy - 1]
            if not (np.isfinite(o_t) and np.isfinite(o_tm1)):
                continue
            rows.append({"rid": rid, "year": yy,
                         "occ_t": o_t, "occ_tm1": o_tm1,
                         "n_t": wide.loc[rid, yy],
                         "n_tm1": wide.loc[rid, yy - 1]})
    return pd.DataFrame(rows)

def cluster_boot_p(mdf, nboot=200):
    """P(occ_t | occ_tm1=1) - P(occ_t | occ_tm1=0), bootstrap over routes."""
    if mdf.empty:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan, (np.nan, np.nan)
    a = mdf[mdf.occ_tm1 == 1].occ_t; b = mdf[mdf.occ_tm1 == 0].occ_t
    if len(a) < 30 or len(b) < 30:
        return np.nan, np.nan, np.nan, np.nan, len(mdf), np.nan, (np.nan, np.nan)

    eff = a.mean() - b.mean()
    na = np.log1p(mdf[mdf.occ_tm1 == 1].n_t); nb = np.log1p(mdf[mdf.occ_tm1 == 0].n_t)
    rids = mdf.rid.unique()
    by_rid = mdf.set_index("rid")
    bs = []; bs_ab = []
    for _ in range(nboot):
        samp = RNG.choice(rids, len(rids), replace=True)
        g = by_rid.loc[samp]  # repeated labels repeat rows: keeps draw multiplicities
        ga = g[g.occ_tm1 == 1].occ_t; gb = g[g.occ_tm1 == 0].occ_t
        if len(ga) > 20 and len(gb) > 20:
            bs.append(ga.mean() - gb.mean())
            bs_ab.append(np.log1p(g[g.occ_tm1 == 1].n_t).mean()
                         - np.log1p(g[g.occ_tm1 == 0].n_t).mean())
    bs = np.array(bs); bs_ab = np.array(bs_ab)
    p = min(1.0, min((bs <= 0).mean(), (bs >= 0).mean()) * 2)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
    eff_ab = na.mean() - nb.mean()
    loab = np.nanpercentile(bs_ab,[2.5,97.5]) if len(bs_ab) else (np.nan,np.nan)
    return eff, lo, hi, p, len(mdf), eff_ab, (loab if len(bs_ab) else (np.nan,np.nan))

def main():
    p = load_parsed()
    env = pd.read_parquet(DATA / "env_all.parquet")
    sp = pd.read_csv(OUT / "species_inclusion.csv")
    inc = sp[sp.included].AOU.astype(str).tolist()
    EC = ["t_ann", "p_ann", "drought_z", "djf_p", "jun_t"]
    hyst, traj = [], []
    for aou in inc:
        d = p[p.AOU == aou][["rid", "Year", "n"]]
        d = d.merge(env, left_on=["rid", "Year"], right_on=["rid", "year"])
        wide = d.pivot_table(index="rid", columns="Year", values="n", aggfunc="sum")
        wide = fill_surveyed_zero(wide)
        E = {c: d.pivot_table(index="rid", columns="Year", values=c) for c in EC}
        Ez = z_anom_base(E)
        mdf = match_effect(wide, Ez, EC)
        eff, lo, hi, pv, n, eff_ab, ci_ab = cluster_boot_p(mdf)
        bal = mdf.groupby("occ_tm1").size().to_dict() if not mdf.empty else {}
        hyst.append({"species": aou, "matched_sets": len(mdf),
                     "effect_of_prior_state": eff, "ci_lo": lo, "ci_hi": hi,
                     "p_boot": pv, "eff_abundance": eff_ab,
                     "ci_ab_lo": ci_ab[0], "ci_ab_hi": ci_ab[1],
                     "n_prior0": bal.get(0.0, 0), "n_prior1": bal.get(1.0, 0)})
        # trajectory: current env near median (|z|<0.5), split by dE sign
        occ = occurrence(wide)
        t = Ez["t_ann"]
        up_eff, dn_eff = [], []
        for rid in t.index:
            zy = t.loc[rid]
            near = zy.index[zy.abs() < 0.5]
            for yy in near:
                if yy - 1 not in t.columns:
                    continue
                dz = zy.get(yy, np.nan) - zy.get(yy - 1, np.nan)
                if not np.isfinite(dz) or abs(dz) < 0.2:
                    continue
                oc = occ.loc[rid, yy]; ocm = occ.loc[rid, yy - 1]
                (up_eff if dz > 0 else dn_eff).append(oc - ocm)
        traj.append({"species": aou,
                     "d_occ_warming": np.nanmean(up_eff) if up_eff else np.nan,
                     "d_occ_cooling": np.nanmean(dn_eff) if dn_eff else np.nan,
                     "n_warm": len(up_eff), "n_cool": len(dn_eff)})
        print(aou, f"hyst_eff={eff:.3f}" if np.isfinite(eff) else "hyst=NA", flush=True)
    pd.DataFrame(hyst).to_csv(OUT / "matched_environment_hysteresis.csv", index=False)
    pd.DataFrame(traj).to_csv(OUT / "trajectory_direction.csv", index=False)

if __name__ == "__main__":
    main()
