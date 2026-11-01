"""Secondary validation per spec: LOYO alongside forward-chaining.
Writes loyo_hg.csv: per-species HG under LOYO (route-demeaned y)."""
import numpy as np, pandas as pd
from core9 import *

EC = ["t_ann", "p_ann", "drought_z", "djf_p", "jun_t"]

def main():
    p = load_parsed()
    env = pd.read_parquet(DATA / "env_all.parquet")
    sp = pd.read_csv(OUT / "species_inclusion.csv")
    inc = sp[sp.included].AOU.astype(str).tolist()
    rows = []
    for aou in inc:
        d = p[p.AOU == aou][["rid", "Year", "n"]]
        d = d.merge(env, left_on=["rid", "Year"], right_on=["rid", "year"])
        if d.empty:
            continue
        wide = d.pivot_table(index="rid", columns="Year", values="n",
                             aggfunc="sum")
        wide = fill_surveyed_zero(wide)
        E = {c: d.pivot_table(index="rid", columns="Year", values=c)
             for c in EC}
        Ez = z_anom_base(E)
        envf = [Ez[c] for c in EC]
        y3, X3, R3, T3 = panel_rows(wide, envf, [1])
        Xe = X3[:, :len(EC)]
        if len(y3):
            yd1, pr1 = loyo_pred(y3, Xe, R3, T3, demean=True)
            yd3, pr3 = loyo_pred(y3, X3, R3, T3, demean=True)
            m1 = r2_from_pred(yd1, pr1)
            m3 = r2_from_pred(yd3, pr3)
        else:
            m1 = m3 = np.nan
        rows.append({"species": aou, "M1_loyo": m1, "M3_loyo": m3,
                     "HG_loyo": m3 - m1})
        print(aou, rows[-1]["HG_loyo"], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "loyo_hg.csv", index=False)

if __name__ == "__main__":
    main()
