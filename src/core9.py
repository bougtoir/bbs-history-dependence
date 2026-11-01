"""Shared utilities for Phase-9 BBS hysteresis analysis.

Primary inference: forward-chaining (train years <= t, predict t+1).
All scaling/demeaning parameters are estimated on train folds only
(no future information in preprocessing).
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bbs"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

_SURV = None

def load_parsed():
    p = pd.read_parquet(DATA / "bbs_parsed.parquet")
    return p[p.RunType == 1]

def survey_mask(ref=None):
    """rid x year bool frame of surveyed route-years (any species record).
    Cached globally; reindexed to `ref` frame if given."""
    global _SURV
    if _SURV is None:
        p = load_parsed()
        _SURV = p[["rid", "Year"]].drop_duplicates().assign(v=True) \
            .pivot_table(index="rid", columns="Year", values="v").notna()
    if ref is None:
        return _SURV
    return _SURV.reindex(index=ref.index, columns=ref.columns).fillna(False)

def occurrence(wide):
    """Occupancy: 1 detected, 0 surveyed-not-detected, NaN not surveyed."""
    S = survey_mask(wide)
    occ = (wide > 0).astype(float)
    return occ.where(S)

def fill_surveyed_zero(wide):
    """Surveyed-but-absent route-years get abundance 0; unsurveyed stay NaN."""
    S = survey_mask(wide)
    return wide.mask(S & wide.isna(), 0.0)

def inclusion_table(p, min_routes=1000, min_years=30, min_nz_frac=0.30):
    S = survey_mask()
    base = p.groupby("AOU").agg(routes=("rid", "nunique"),
                              years=("Year", "nunique"),
                              nz=("n", lambda s: (s > 0).sum()),
                              med_n=("n", "median")).reset_index()
    # denominator: surveyed route-years on routes where the species occurs
    rts = p.groupby("AOU").rid.unique()
    tot = {a: int(S.loc[S.index.intersection(r)].values.sum())
           for a, r in rts.items()}
    base["tot"] = base.AOU.map(tot)
    sp = base
    sp["nz_frac"] = sp.nz / sp.tot
    inc = sp[(sp.routes >= min_routes) & (sp.years >= min_years)
             & (sp.nz_frac >= min_nz_frac)]
    sp["included"] = sp.AOU.isin(inc.AOU)
    sp["reason"] = ""
    sp.loc[sp.routes < min_routes, "reason"] = "few_routes"
    sp.loc[(sp.routes >= min_routes) & (sp.years < min_years),
           "reason"] = "few_years"
    sp.loc[(sp.routes >= min_routes) & (sp.years >= min_years)
           & (sp.nz_frac < min_nz_frac), "reason"] = "low_prevalence"
    return sp

def species_matrices(p, env, aou, envcols):
    d = p[p.AOU == aou][["rid", "Year", "n"]]
    d = d.merge(env, left_on=["rid", "Year"], right_on=["rid", "year"])
    E = {c: d.pivot_table(index="rid", columns="Year", values=c)
         for c in envcols}
    wide = d.pivot_table(index="rid", columns="Year", values="n",
                         aggfunc="sum")
    wide = fill_surveyed_zero(wide)
    return wide, E

def z_anom_base(E, cutoff=2005):
    """Route-wise z-anomaly using stats from years <= cutoff only."""
    out = {}
    for k, fr in E.items():
        base = fr[[c for c in fr.columns if c <= cutoff]]
        mu, sd = base.mean(1), base.std(1)
        out[k] = fr.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return out

def r2_from_pred(y, pred):
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 20 or np.sum(y[ok] ** 2) == 0:
        return np.nan
    return 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum(y[ok] ** 2)

def _fit(Xtr, ytr, Xte):
    mu = Xtr.mean(0) if Xtr.shape[1] else np.zeros(0)
    A = np.c_[np.ones(len(Xtr)), Xtr - mu]
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.c_[np.ones(len(Xte)), Xte - mu] @ beta

def _route_mean(y, R):
    s = pd.Series(y)
    return s.groupby(R).transform("mean").values

def _demean(y, R, tr):
    """route mean estimated on train rows only, applied to all rows."""
    s = pd.Series(y[tr])
    mu_map = s.groupby(R[tr]).mean()
    gm = np.nanmean(y[tr])
    return y - np.array([mu_map.get(r, gm) for r in R])

def forward_chain(y, X, R, T, demean=False, window=None, min_hist=1):
    """train years<=t (last `window` years if set), predict t+1.
    Returns (y_eval, pred). demean: route-mean subtracted, train-only."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    pred = np.full(len(y), np.nan); yev = np.full(len(y), np.nan)
    uy = np.unique(T)
    for i, t in enumerate(uy[:-1]):
        lo = uy[max(0, i - (window - 1))] if window else uy[0]
        tr = (T <= t) & (T >= lo); te = T == t + 1
        if tr.sum() < X.shape[1] + 5 or te.sum() == 0:
            continue
        if len(np.unique(T[tr])) < min_hist:
            continue
        if demean:
            yd = _demean(y, R, tr)
        else:
            yd = y
        pred[te] = _fit(X[tr], yd[tr], X[te])
        yev[te] = yd[te]
    return yev, pred

def loyo_pred(y, X, R, T, demean=False, year_demean=False):
    y = np.asarray(y, float); X = np.asarray(X, float)
    pred = np.full(len(y), np.nan); yd = np.full(len(y), np.nan)
    for t in np.unique(T):
        tr, te = T != t, T == t
        if tr.sum() < X.shape[1] + 5 or te.sum() == 0:
            continue
        if demean:
            yw = _demean(y, R, tr)
        elif year_demean:
            # Phase-8B style: subtract year mean across routes.
            # train year-means from train only; held-out year mean from its
            # own rows (retrospective, matches 8B).
            yw = y.copy()
            ym_tr = pd.Series(y[tr]).groupby(T[tr]).transform("mean")
            yw[tr] = y[tr] - ym_tr.values
            yw[te] = y[te] - np.nanmean(y[te])
        else:
            yw = y
        pred[te] = _fit(X[tr], yw[tr], X[te])
        yd[te] = yw[te]
    return yd, pred

def panel_rows(wide, E_lagged, hist_lags):
    Y, X, R, T = [], [], [], []
    cols = wide.columns
    for rid in wide.index:
        for yy in cols:
            row = [fr.loc[rid, yy] if (rid in fr.index and yy in fr.columns)
                   else np.nan for fr in E_lagged]
            for h in hist_lags:
                row.append(wide.loc[rid, yy - h]
                           if (yy - h) in cols else np.nan)
            Y.append(wide.loc[rid, yy]); X.append(row); R.append(rid); T.append(yy)
    if len(Y) == 0:
        return np.array([]), np.zeros((0, 0)), np.array([]), np.array([])
    y = np.asarray(Y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    R = np.asarray(R); T = np.asarray(T)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    return y[ok], X[ok], R[ok], T[ok]
