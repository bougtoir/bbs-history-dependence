"""Fetch NASA POWER monthly climate for all BBS routes used by included species.
Outputs per-route annual aggregates: t_ann, jun_t, p_ann, djf_p, drought_z.
Resumable via data/env_chunks/<rid>.parquet."""
import json, time, urllib.request
import numpy as np, pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "bbs"
CH = Path(__file__).resolve().parents[1] / "data" / "env_chunks"
CH.mkdir(parents=True, exist_ok=True)

def _get(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as ex:
            wait = min(2 ** k * 2, 60)
            print("retry", k, ex, flush=True)
            time.sleep(wait)
    raise RuntimeError("fail " + url)

def om_daily(lat, lon, y0=1990, y1=2024):
    """Open-Meteo daily archive -> same annual aggregates."""
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date={y0}-01-01"
           f"&end_date={y1}-12-31"
           "&daily=temperature_2m_mean,precipitation_sum,snowfall_sum"
           "&timezone=UTC")
    d = _get(url)["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"]),
                       "t": d["temperature_2m_mean"],
                       "p": d["precipitation_sum"],
                       "snow": d["snowfall_sum"]})
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    ann = df.groupby("year").agg(t_ann=("t", "mean"), p_ann=("p", "sum"),
                                 jun_t=("t", lambda s: s[df.loc[s.index, "month"] == 6].mean()),
                                 jun_p=("p", lambda s: s[df.loc[s.index, "month"] == 6].sum()))
    dd = df[df.month.isin([12, 1, 2])].copy()
    dd["wyear"] = dd.year + (dd.month == 12).astype(int)
    djf = dd.groupby("wyear").snow.sum().rename("djf_p")
    return ann.join(djf, on="year").reset_index()

def power_monthly(lat, lon, y0=1990, y1=2024):
    try:
        return om_daily(lat, lon, y0, y1)
    except Exception:
        pass
    url = ("https://power.larc.nasa.gov/api/temporal/monthly/point?"
           "parameters=T2M,PRECTOTCORR&community=AG&"
           f"longitude={lon}&latitude={lat}&start={y0}&end={y1}&format=JSON")
    p = _get(url)["properties"]["parameter"]
    ks = [k for k in p["T2M"] if 1 <= int(k[4:]) <= 12]
    ks.sort()
    get = lambda d, k: np.nan if d.get(k, -999) <= -900 else d[k]
    df = pd.DataFrame({"ym": pd.PeriodIndex(ks, freq="M"),
                       "t": [get(p["T2M"], k) for k in ks],
                       "p": [get(p["PRECTOTCORR"], k) for k in ks]})
    df["year"] = df.ym.dt.year
    df["month"] = df.ym.dt.month
    ann = df.groupby("year").agg(t_ann=("t", "mean"), p_ann=("p", "sum"),
                                 jun_t=("t", lambda s: s[df.loc[s.index, "month"] == 6].mean()),
                                 jun_p=("p", lambda s: s[df.loc[s.index, "month"] == 6].sum()))
    d = df[df.month.isin([12, 1, 2])].copy()
    d["wyear"] = d.year + (d.month == 12).astype(int)
    djf = d.groupby("wyear").p.sum().rename("djf_p")
    return ann.join(djf, on="year").reset_index()

def main():
    rc = pd.read_csv(DATA / "route_coords.csv").set_index("rid")
    p = pd.read_parquet(DATA / "bbs_parsed.parquet")
    p = p[p.RunType == 1]
    yrs = p.groupby("rid").Year.nunique()
    routes = yrs[yrs >= 10].index
    rows, t0 = [], time.time()
    for i, rid in enumerate(routes):
        cf = CH / f"{rid}.parquet"
        if cf.exists():
            continue
        if rid not in rc.index:
            continue
        lat, lon = rc.loc[rid, ["Latitude", "Longitude"]]
        try:
            e = power_monthly(lat, lon)
        except Exception as ex:
            print("fail", rid, ex, flush=True); continue
        e["rid"] = rid
        e.to_parquet(cf)
        if i % 100 == 0:
            print(i, "of", len(routes), f"{time.time()-t0:.0f}s", flush=True); time.sleep(0.6)
    # concat
    env = pd.concat([pd.read_parquet(f) for f in CH.glob("*.parquet")],
                    ignore_index=True)
    env["drought_z"] = env.groupby("rid").p_ann.transform(
        lambda s: (s - s.mean()) / s.std())
    env.to_parquet(DATA / "env_all.parquet")
    print("env rows:", len(env))

if __name__ == "__main__":
    main()
