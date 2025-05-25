import os
import pandas as pd
from csv_handler import CSVHandler


def load_data():
    """
    Read all parameter CSV files from the project root's data_folder and return (sets, params).

    Expects files in <project_root>/data_folder:
      - demand.csv   (i, n, t, D)
      - holding.csv  (i, n, h)
      - capacity.csv (n, C)
      - distance.csv (n_from, n_to, d)
      - alpha.csv    (i, alpha)
      - optional beta.csv (beta)

    Returns:
        sets: dict of sets for LPSolver
        params: dict of parameters for LPSolver, including 'max_perishable_age'
    """
    # project-root/data_folder
    data_dir = os.path.join(os.getcwd(), "data_folder")

    def read_csv(fname):
        path = os.path.join(data_dir, fname)
        df = CSVHandler(path).read()
        if df is None:
            raise FileNotFoundError(f"Failed to read {path}")
        return df

    # Load dataframes
    demand_df = read_csv("demand.csv")
    holding_df = read_csv("holding.csv")
    cap_df = read_csv("capacity.csv")
    dist_df = read_csv("distance.csv")
    alpha_df = read_csv("alpha.csv")

    # Optional beta
    beta_path = os.path.join(data_dir, "beta.csv")
    if os.path.exists(beta_path):
        beta = float(CSVHandler(beta_path).read()["beta"].iloc[0])
    else:
        beta = 0.1

    # Read max perishable age from env or default
    max_perishable_age = 3

    # Build param dicts
    D = {(row.i, row.n, int(row.t)): float(row.D) for row in demand_df.itertuples()}
    h = {(row.i, row.n): float(row.h) for row in holding_df.itertuples()}
    C = {row.n: float(row.C) for row in cap_df.itertuples()}
    d = {(row.n_from, row.n_to): float(row.d) for row in dist_df.itertuples()}
    alpha = {row.i: float(row.alpha) for row in alpha_df.itertuples()}

    # Define sets
    T = sorted(demand_df["t"].astype(int).unique())
    P = sorted(holding_df["i"].unique())
    NP = sorted(set(alpha_df["i"]) - set(P)) + sorted(set(alpha_df["i"]) & set(P))
    all_arcs = list(d.keys())
    ws_arcs = [(u, v) for (u, v) in all_arcs if u.startswith("W") and v.startswith("S")]
    nodes_F = sorted(
        dist_df.loc[dist_df["n_from"].str.startswith("F"), "n_from"].unique()
    )
    nodes_WS = sorted(set(dist_df["n_from"]).union(dist_df["n_to"]))
    nodes_S = sorted(dist_df.loc[dist_df["n_to"].str.startswith("S"), "n_to"].unique())
    A_P = list(range(1, max_perishable_age + 1))

    sets = {
        "T": T,
        "P": P,
        "NP": NP,
        "A_P": A_P,
        "all_arcs": all_arcs,
        "ws_arcs": ws_arcs,
        "nodes_F": nodes_F,
        "nodes_WS": nodes_WS,
        "nodes_S": nodes_S,
    }

    params = {
        "D": D,
        "h": h,
        "U": 100.0,
        "C": C,
        "d": d,
        "alpha": alpha,
        "beta": beta,
        "max_perishable_age": max_perishable_age,
    }

    print(f"[INFO] Loaded data from {data_dir} (max_age={max_perishable_age})")
    return sets, params
