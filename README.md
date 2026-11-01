# Disentangling spatial persistence from temporal history in continental bird distributions

Analysis code and canonical result tables for the study decomposing lagged-state
predictability in the North American Breeding Bird Survey (BBS) into persistent
spatial structure vs genuine temporal history (71,694 surveyed route-years,
116 species, 1966–2023).

## Data sources (public)

- **BBS**, 2024 data release covering 1966–2023, USGS ScienceBase item
  `66d9ed16d34eef5af66d534b` (Routes.csv, Weather.csv, 50-StopData, SpeciesList.csv).
- **Climate**: NASA POWER monthly API (T2M, PRECTOTCORR) fetched per route
  (`src/env_fetch.py`).

Raw files are not redistributed here; download them from the sources above.

## Pipeline

```
src/bbs_parse.py            # Routes + Weather + stop50 -> bbs_parsed.parquet (RunType==1)
src/env_fetch.py            # per-route monthly climate -> env_all.parquet
src/species_hg.py           # 2x2 design: LOYO x forward-chaining x route demeaning -> HG tables
src/matched.py              # matched-environment occupancy/abundance contrasts
src/anomaly.py              # climate-anomaly redistribution
src/sim9.py                 # false-positive / power simulations
src/observers_mod.py        # observer covariates + state-level regional aggregation
src/loyo_compare.py         # leave-one-year-out comparator
src/audit_observer_persist.py  # observer-turnover abundance contrast
src/figures9.py, report9.py # figures and reports
```

All models are unregularized linear least squares (`np.linalg.lstsq`) with an
intercept and training-mean centering. Out-of-sample R² cells are NA when
SST_test = 0 or SSE/SST_test > 1e6 (numerically singular fits).

## Results

`results/tables/` contains the canonical CSV outputs the manuscript values are
computed from (species-level history gains, matched-environment contrasts,
anomaly redistribution, observer controls, simulations).

## Manuscript

Submitted to a journal; submission files are maintained in the private working
repository. Corresponding values in the text are generated dynamically from
`results/tables/`.
