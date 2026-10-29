import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def make_simulation_map(base_dir="datacloud/digi_giulia_fe_zcone_zsteps",output_file="simulation_map.yaml"):
    alphas  = np.linspace(0.019,0.023,11)
    lambdas = np.linspace(850,1850,11)

    zpos_dic = {"5": "5.0", "15": "15.0", "25": "25.0", "35": "35.0", "46p5": "46.5"}

    sim_dict = defaultdict(list)
    base_dir = Path(base_dir)
    
    for a, alpha in enumerate(alphas):
        for l, Lambda in enumerate(lambdas):
            for z_key, z_val in zpos_dic.items():
                # Costruisce il percorso fino a: .../z15cm_dr/digi_3-5
                digi_dir = base_dir / f"z{z_key}cm_dr" / f"digi_{a}-{l}"
                
                if not digi_dir.is_dir():
                    continue
                print(f"Looking for files corresponding to alpha={alpha}, lambda={Lambda}, z={z_val}")
                
                # Cerca in tutte le cartelle job_0, job_1, ...
                for job_dir in digi_dir.glob("job_*"):
                    if job_dir.is_dir():
                        filename = job_dir / "reco_run00001_3D.root"
                        
                        if filename.exists():
                            # Controllo dimensione del file in KB (> 100 KB)
                            size_kb = filename.stat().st_size / 1024
                            if size_kb > 100:
                                dict_key = f"{z_val},{alpha:.4f},{Lambda:.0f}"
                                sim_dict[dict_key].append(str(filename))
    
    with open(output_file, "w") as f:
        yaml.safe_dump(dict(sim_dict), f, sort_keys=True, default_flow_style=False)
    print (f"===> Output sim map written into {output_file}")


def make_data_map(input_csv="data_reading/calibration.csv",output_file="data_map.yaml"):
    df = pd.read_csv(input_csv, sep=",")

    nP = nT = nH = 5
    
    z_vals = df["z"].unique()
    P_bins = np.linspace(df["P"].min(), df["P"].max(), nP + 1)
    T_bins = np.linspace(df["T"].min(), df["T"].max(), nT + 1)
    H_bins = np.linspace(df["H"].min(), df["H"].max(), nT + 1)

    nz = len(z_vals)
    print(f"Now will choose a set of {nz} points in z, {nP} points in P, {nT} points in T and {nH} points in H, sampling uniformly from their min/max values")
    
    df = df.copy()

    
    df["z_bin"] = (df[df["z"].isin(z_vals)])["z"]
    df["P_bin"] = pd.cut(df["P"], P_bins, include_lowest=True)
    df["T_bin"] = pd.cut(df["T"], T_bins, include_lowest=True)
    df["H_bin"] = pd.cut(df["H"], H_bins, include_lowest=True)

    grouped = (
        df
        .dropna(subset=["z_bin", "P_bin", "T_bin", "H_bin"])
        .groupby(["z_bin", "P_bin", "T_bin", "H_bin"], observed=True)["run"]
        .apply(list)
    )

    print(f"In principle I could select {nz*nP*nT*nH} points, but some bins can be empty, so I selected instead {len(grouped)} combinations of z, P, T, H.")

    data_dict = {
        (float(z), float(round(P.mid,3)), float(round(T.mid,1)), float(round(H.mid,1))): [f"datacloud/runs/recodata_run4/reco_run{int(r)}_3D.root" for r in runs]
        for (z, P, T, H), runs in grouped.items()
    }

    data_dict_simplekeys = {", ".join(map(str, k)): v for k, v in data_dict.items()}
    
    with open(output_file, "w") as f:
        yaml.safe_dump(data_dict_simplekeys, f, sort_keys=True, default_flow_style=False)
    print (f"===> Output data map written into {output_file}")

    
if __name__ == "__main__":

    make_simulation_map()
    make_data_map()
    
