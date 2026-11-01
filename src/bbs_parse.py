"""Parse BBS 50-stop files -> route-year total count per species + effort join.
Then fetch June climate (ERA5 via open-meteo) per route for env predictors."""
import glob, json, time, urllib.request
import numpy as np, pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "bbs"

def main():
    routes = pd.read_csv(DATA / "Routes.csv", encoding="latin1")
    routes["rid"] = (routes.CountryNum.astype(str) + "-" + routes.StateNum.astype(str)
                     + "-" + routes.Route.astype(str))
    stop_cols = [f"Stop{i}" for i in range(1, 51)]
    frames = []
    for f in sorted(glob.glob(str(DATA / "stop50" / "*.csv"))):
        d = pd.read_csv(f, dtype={c: str for c in stop_cols}, encoding="latin1")
        d["rid"] = (d.CountryNum.astype(str) + "-" + d.StateNum.astype(str)
                    + "-" + d.Route.astype(str))
        d["AOU"] = d.AOU.astype(str)
        d["n"] = d[stop_cols].apply(pd.to_numeric, errors="coerce").sum(1)
        frames.append(d[["rid", "Year", "AOU", "n", "RouteDataID"]])
    bbs = pd.concat(frames, ignore_index=True)
    wea = pd.read_csv(DATA / "Weather.csv", dtype=str, encoding="latin1")
    wea["rid"] = (wea.CountryNum.astype(int).astype(str) + "-"
                  + wea.StateNum.astype(int).astype(str) + "-"
                  + wea.Route.astype(int).astype(str))
    wea["Year"] = wea.Year.astype(int)
    wea["RunType"] = pd.to_numeric(wea.RunType, errors="coerce")
    wea["obs"] = wea.ObsN.astype(str).str.strip()
    wea["StartTempF"] = pd.to_numeric(wea.StartTemp, errors="coerce")
    bbs = bbs.merge(wea[["rid", "Year", "RunType", "obs", "StartTempF"]],
                    on=["rid", "Year"], how="left", suffixes=("", "_w"))
    bbs = bbs[bbs.RunType == 1]  # acceptable-quality runs only
    # species selection: top 10 by number of route-years with presence
    prev = bbs[bbs.n > 0].groupby("AOU").size().sort_values(ascending=False)
    top10 = prev.head(10).index.tolist()
    pd.Series(top10).to_csv(DATA / "bbs_top_species.csv", index=False)
    bbs.to_parquet(DATA / "bbs_parsed.parquet", index=False)
    print("rows", len(bbs), "top species:", top10)
    # route coords for env fetch
    rc = routes.drop_duplicates("rid")[["rid", "Latitude", "Longitude"]]
    rc.to_csv(DATA / "route_coords.csv", index=False)
    print("routes:", len(rc))

if __name__ == "__main__":
    main()
