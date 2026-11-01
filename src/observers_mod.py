"""Observer controls (ObsN first-year flag, RPID protocol), occurrence
robustness, regional aggregation, and strategy moderators (migratory vs
resident, hand-coded external classification)."""
import numpy as np, pandas as pd
from core9 import *

EC = ["t_ann", "p_ann", "drought_z", "djf_p", "jun_t"]

# independent migratory classification (Sibley/eBird-style):
# LD = long-distance (Neotropical) migrant, SD = short-distance, R = resident
def load_migration(sp_aous):
    sl = pd.read_csv(DATA / "SpeciesList.csv")
    fam_mig = {  # family-level coarse defaults; overridden by LD set
        "LD": {"Parulidae", "Vireonidae", "Tyrannidae", "Trochilidae",
               "Icteridae", "Cardinalidae", "Cathartidae", "Scolopacidae",
               "Laridae", "Caprimulgidae", "Apodidae", "Hirundinidae",
               "Polioptilidae"},
        "R": {"Odontophoridae", "Phasianidae", "Corvidae", "Paridae",
              "Sittidae", "Picidae", "Strigidae", "Accipitridae",
              "Falconidae", "Troglodytidae", "Cardinalidae",
              "Fringillidae", "Columbidae"}}
    sl["AOU"] = sl.AOU.astype(str)
    fam2aou = sl.set_index("AOU")["Family"].to_dict()
    name = sl.set_index("AOU")["English_Common_Name"].to_dict()
    out = {}
    for a in sp_aous:
        f = fam2aou.get(str(a), "")
        if f in fam_mig["LD"]:
            out[a] = "migrant"
        elif f in fam_mig["R"]:
            out[a] = "resident"
        else:
            out[a] = "short_dist"
    return pd.DataFrame({"species": list(out), "mig": list(out.values()),
                         "name": [name.get(a) for a in out]})

def main():
    p = load_parsed()
    env = pd.read_parquet(DATA / "env_all.parquet")
    sp = pd.read_csv(OUT / "species_inclusion.csv")
    inc = sp[sp.included].AOU.astype(str).tolist()
    hg_tbl = pd.read_csv(OUT / "bbs_species_history_gain.csv")
    hg_tbl["species"] = hg_tbl.species.astype(str)
    mig = load_migration(inc)
    # first-year-observer flag per route-year (observer's first yr on route)
    w = pd.read_csv(DATA / "Weather.csv",
                    usecols=["RouteDataID", "ObsN", "Year", "RPID"])
    oy = w.groupby(["RouteDataID", "ObsN"]).Year.min().rename("first_yr")
    w = w.merge(oy, on=["RouteDataID", "ObsN"], how="left")
    w["obs_new"] = (w.Year == w.first_yr).astype(int)
    pp = p.merge(w[["RouteDataID", "obs_new", "RPID"]],
                 on="RouteDataID", how="left")
    res = []
    for aou in inc:
        d = pp[pp.AOU == aou][["rid", "Year", "n", "obs_new"]]
        d = d.merge(env, left_on=["rid", "Year"], right_on=["rid", "year"])
        wide = d.pivot_table(index="rid", columns="Year", values="n", aggfunc="sum")
        wide = fill_surveyed_zero(wide)
        occ = occurrence(wide)
        E = {c: d.pivot_table(index="rid", columns="Year", values=c) for c in EC}
        Ez = z_anom_base(E)
        envf = [Ez[c] for c in EC]
        obsF = d.pivot_table(index="rid", columns="Year", values="obs_new")
        # abundance models with observer covariate; all evaluated on one
        # common panel (rows finite for env + obs covariate + lag-1 history)
        nE = len(EC)
        yA, XA, RA, TA = panel_rows(wide, envf + [obsF], [1])
        def fc_slice(Xc, yA=yA, RA=RA, TA=TA):
            if len(yA) == 0:
                return np.nan
            ye, pr = forward_chain(yA, Xc, RA, TA, demean=True)
            return r2_from_pred(ye, pr)
        Xe = XA[:, :nE] if len(yA) else np.zeros((0, 0))
        Xeo = XA[:, :nE + 1] if len(yA) else np.zeros((0, 0))
        Xh = XA[:, -1:]
        r = {"species": aou,
             "M1_obs": fc_slice(Xeo),
             "M3_obs": fc_slice(np.c_[Xeo, Xh]) if len(yA) else np.nan,
             "M1": fc_slice(Xe), "M3": fc_slice(np.c_[Xe, Xh]) if len(yA)
                    else np.nan}
        # occurrence model on the common (masked) occupancy panel
        yo3, Xo3, Ro3, To3 = panel_rows(occ, envf, [1])
        if len(yo3):
            Xoe = Xo3[:, :nE]
            ye_, po3 = forward_chain(yo3, Xo3, Ro3, To3, demean=True)
            ye0, po = forward_chain(yo3, Xoe, Ro3, To3, demean=True)
            r["HG_occ"] = r2_from_pred(ye_, po3) - r2_from_pred(ye0, po)
        else:
            r["HG_occ"] = np.nan
        # regional aggregation: state (Country-State prefix)
        reg = d.copy()
        reg["reg"] = reg.rid.str.split("-").str[0]  # state/country code
        rgd = reg.groupby(["reg", "Year"]).n.mean().reset_index()
        rw = rgd.pivot_table(index="reg", columns="Year", values="n")
        env_r = env.copy()
        env_r["reg"] = env_r.rid.str.split("-").str[0]
        re_ = rgd.merge(env_r.groupby(["reg", "year"])[EC].mean()
                        .reset_index(),
                        left_on=["reg", "Year"], right_on=["reg", "year"],
                        how="left")
        rE = {c: re_.pivot_table(index="reg", columns="Year", values=c)
              for c in EC}
        rEz = z_anom_base(rE)
        yg3, Xg3, Rg3, Tg3 = panel_rows(rw, [rEz[c] for c in EC], [1])
        if len(yg3):
            Xge = Xg3[:, :len(EC)]
            yge3, pg3 = forward_chain(yg3, Xg3, Rg3, Tg3, demean=True)
            yge, pg = forward_chain(yg3, Xge, Rg3, Tg3, demean=True)
            r["HG_regional"] = r2_from_pred(yge3, pg3) - r2_from_pred(yge, pg)
        else:
            r["HG_regional"] = np.nan
        r["HG_abund"] = r["M3"] - r["M1"]
        res.append(r)
        print(aou, flush=True)
    df = pd.DataFrame(res).merge(mig, on="species", how="left")
    df = df.merge(hg_tbl[["species", "HG"]], on="species", how="left")
    df.to_csv(OUT / "observer_occurrence_scale_moderators.csv", index=False)
    print(df.groupby("mig").HG.describe())

if __name__ == "__main__":
    main()
