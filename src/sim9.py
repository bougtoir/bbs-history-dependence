"""Observation-structure simulations A-H on real BBS masks.

Scenarios:
 A env-only tracking        : y ~ env z
 B delayed env response     : y ~ env + env_{t-1}
 C static route preference  : y ~ route random effect only
 D observation persistence  : env + AR(1) observation noise
 E true history dependence  : y ~ env + 0.5*y_{t-1}
 F true hysteresis          : occupancy persists conditional on env
 G missingness/observer turnover: A + random missingness
 H abundance autocorrelation: AR(1) on abundance residual (not occupancy)

Pipeline metric: forward-chaining HG and matched-env prior-state effect.
Required: P(false hysteresis | A)<=0.10, P(false hysteresis | C)<=0.10.
"""
import numpy as np, pandas as pd
from core9 import *
from matched import match_effect

RNG = np.random.default_rng(2)
EC = ["t_ann", "p_ann", "drought_z", "djf_p", "jun_t"]
NREP = 10

def simulate(wide, Ez, kind):
    cols, idx = wide.columns, wide.index
    y = pd.DataFrame(0.0, index=idx, columns=cols)
    base_env = sum(Ez[c].fillna(0) for c in EC) / len(EC)
    if kind == "A":
        y = base_env + pd.DataFrame(RNG.normal(0, 1, wide.shape),
                                    index=idx, columns=cols)
    elif kind == "B":
        y = 0.6 * base_env + 0.4 * base_env.shift(axis=1)
        y = y + pd.DataFrame(RNG.normal(0, 1, wide.shape), index=idx, columns=cols)
    elif kind == "C":
        re = RNG.normal(0, 1.5, len(idx))
        y = pd.DataFrame(np.tile(re[:, None], (1, len(cols)))
                         + RNG.normal(0, 0.5, wide.shape),
                         index=idx, columns=cols)
    elif kind == "D":
        e = pd.DataFrame(RNG.normal(0, 1, wide.shape), index=idx, columns=cols)
        ar = pd.DataFrame(0.0, index=idx, columns=cols)
        prev = np.zeros(len(idx))
        for c in cols:
            prev = 0.7 * prev + e[c].values
            ar[c] = prev
        y = base_env.fillna(0) + ar
    elif kind == "E":
        e = pd.DataFrame(RNG.normal(0, 1, wide.shape), index=idx, columns=cols)
        prev = np.zeros(len(idx))
        for c in cols:
            cur = base_env[c].fillna(0).values + 0.5 * prev + e[c].values
            y[c] = cur; prev = cur
    elif kind == "F":
        e = pd.DataFrame(RNG.normal(0, 0.5, wide.shape), index=idx, columns=cols)
        occ_prev = np.zeros(len(idx))
        for c in cols:
            p_env = 1 / (1 + np.exp(-base_env[c].fillna(0).values))
            occ = np.where(occ_prev == 1,
                           np.clip(p_env + 0.45, 0, 1), np.clip(p_env, 0, 1))
            occ_prev = (RNG.random(len(idx)) < occ).astype(float)
            y[c] = occ_prev * 10 + e[c].values
    elif kind == "G":
        y = simulate(wide, Ez, "A")
        miss = RNG.random(wide.shape) < 0.3
        y = y.mask(miss)
    elif kind == "H":
        e = pd.DataFrame(RNG.normal(0, 1, wide.shape), index=idx, columns=cols)
        prev = np.zeros(len(idx))
        for c in cols:
            prev = 0.6 * prev + e[c].values
            y[c] = base_env[c].fillna(0).values + prev
    return y

def pipeline_metrics(y, Ez):
    """forward-chaining HG + matched-env prior effect on simulated frame."""
    envf = [Ez[c] for c in EC]
    yy3, X3, R3, T3 = panel_rows(y, envf, [1])
    if len(yy3) == 0:
        return np.nan, np.nan
    Xe = X3[:, :len(EC)]
    ye1, pred1 = forward_chain(yy3, Xe, R3, T3, demean=True)
    ye3, pred3 = forward_chain(yy3, X3, R3, T3, demean=True)
    hg = r2_from_pred(ye3, pred3) - r2_from_pred(ye1, pred1)
    m = match_effect(y, Ez, EC)
    a = m[m.occ_tm1 == 1].occ_t; b = m[m.occ_tm1 == 0].occ_t
    hyst = (a.mean() - b.mean()) if len(a) > 30 and len(b) > 30 else np.nan
    return hg, hyst

def main():
    p = load_parsed()
    env = pd.read_parquet(DATA / "env_all.parquet")
    sp = pd.read_csv(OUT / "species_inclusion.csv")
    inc = sp[sp.included].AOU.astype(str).tolist()
    focal = inc[:3]  # prespecified: first 3 by AOU order (sufficient coverage)
    rows = []
    for aou in focal:
        d = p[p.AOU == aou][["rid", "Year", "n"]]
        d = d.merge(env, left_on=["rid", "Year"], right_on=["rid", "year"])
        wide = d.pivot_table(index="rid", columns="Year", values="n", aggfunc="sum")
        E = {c: d.pivot_table(index="rid", columns="Year", values=c) for c in EC}
        Ez = z_anom_base(E)
        for kind in list("ABCDEFGH"):
            hgs, hys = [], []
            for r in range(NREP):
                y = simulate(wide, Ez, kind)
                y = y.where(survey_mask(y), np.nan)  # real survey coverage
                hg, hy = pipeline_metrics(y, Ez)
                hgs.append(hg); hys.append(hy)
            rows.append({"species": aou, "scenario": kind,
                         "P_HG_pos": np.mean(np.array(hgs) > 0),
                         "P_HG_gt_005": np.mean(np.array(hgs) > 0.05),
                         "median_HG": np.nanmedian(hgs),
                         "p95_HG": np.nanpercentile(hgs, 95),
                         "P_hyst_gt_0": np.mean(np.array(hys) > 0),
                         "median_hyst": np.nanmedian(hys)})
            print(aou, kind, rows[-1]["median_HG"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulation_false_positive.csv", index=False)

if __name__ == "__main__":
    main()
