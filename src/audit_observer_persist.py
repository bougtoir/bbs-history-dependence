"""Observer-persistence control for the abundance prior-state effect.

Re-estimates the matched-environment abundance prior-state contrast
(log1p(n_t) | prior present - | prior absent) restricted to consecutive
route-year pairs where the OBSERVER changed (ObsN_{t} != ObsN_{t-1}), and
the complement (same observer). If the +0.63 effect were an
observer-continuity artifact, it should concentrate in same-observer pairs.

Output: geb/results/audit/observer_persistence_abundance.csv
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from core9 import load_parsed, DATA, OUT, species_matrices, z_anom_base
from matched import match_effect
from pathlib import Path

GEB = Path(__file__).resolve().parents[2] / "geb" / "results" / "audit"
EC = ["t_ann", "p_ann", "drought_z", "djf_p", "jun_t"]

def main():
    p = load_parsed()
    env = pd.read_parquet(DATA / "env_all.parquet")
    w = pd.read_csv(DATA / "Weather.csv",
                    usecols=["RouteDataID", "ObsN", "Year"])
    obs_by_ry = w.set_index(["RouteDataID", "Year"]).ObsN
    sp = pd.read_csv(OUT / "species_inclusion.csv")
    inc = sp[sp.included].AOU.astype(str).tolist()
    rows = []
    for aou in inc:
        d = p[p.AOU == aou][["rid", "Year", "n", "RouteDataID"]]
        d = d.merge(env, left_on=["rid", "Year"], right_on=["rid", "year"])
        wide = d.pivot_table(index="rid", columns="Year", values="n", aggfunc="sum")
        from core9 import fill_surveyed_zero
        wide = fill_surveyed_zero(wide)
        # observer id per rid-year via RouteDataID
        rd = p[p.AOU == aou][["rid", "Year", "RouteDataID"]].drop_duplicates()
        E = {c: d.pivot_table(index="rid", columns="Year", values=c) for c in EC}
        Ez = z_anom_base(E)
        mdf = match_effect(wide, Ez, EC)
        if mdf.empty:
            continue
        # map rid-year -> RouteDataID -> ObsN
        rid_rd = (rd.drop_duplicates(["rid", "Year"])
                    .set_index(["rid", "Year"]).RouteDataID.to_dict())
        o_now = mdf.apply(lambda r: obs_by_ry.get(
            (rid_rd.get((r.rid, r.year), None), r.year), np.nan), axis=1)
        o_prev = mdf.apply(lambda r: obs_by_ry.get(
            (rid_rd.get((r.rid, r.year - 1), None), r.year - 1), np.nan), axis=1)
        mdf = mdf.assign(o_now=o_now.values, o_prev=o_prev.values)
        mdf = mdf.dropna(subset=["o_now", "o_prev"])
        mdf["turnover"] = (mdf.o_now != mdf.o_prev).astype(int)
        def eff(sub):
            a = np.log1p(sub[sub.occ_tm1 == 1].n_t)
            b = np.log1p(sub[sub.occ_tm1 == 0].n_t)
            if len(a) < 15 or len(b) < 15:
                return np.nan, len(a), len(b)
            return a.mean() - b.mean(), len(a), len(b)
        e_t, n_t1, n_t0 = eff(mdf[mdf.turnover == 1])
        e_s, n_s1, n_s0 = eff(mdf[mdf.turnover == 0])
        e_a, _, _ = eff(mdf)
        rows.append({"species": aou,
                     "eff_all": e_a,
                     "eff_observer_turnover": e_t,
                     "eff_same_observer": e_s,
                     "n_turnover_prior1": n_t1, "n_turnover_prior0": n_t0,
                     "n_same_prior1": n_s1, "n_same_prior0": n_s0,
                     "share_turnover": mdf.turnover.mean()})
        print(aou, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(GEB / "observer_persistence_abundance.csv", index=False)
    print("median eff_all:", df.eff_all.median(),
          "| turnover:", df.eff_observer_turnover.median(),
          "| same:", df.eff_same_observer.median(),
          "| n defined turnover:", df.eff_observer_turnover.notna().sum())

if __name__ == "__main__":
    main()
