import os
import pandas as pd
from handler.csv_handler import CSVHandler


def load_data():
    """
    Read all parameter CSV files from the project root's data_folder and return (sets, params).

    Expects files in <project_root>/data_folder:
      - demand.csv           (i, n, t, D)
      - CA_holding.csv       (store_id, item_id, holding_cost)
      - distance.csv         (n_from, n_to, d)
      - item.csv             (i)
      - optional beta.csv    (beta)

    Returns:
        sets: dict of sets for LPSolver
        params: dict of parameters for LPSolver
    """
    # project-root/data_folder
    project_root = os.getcwd()
    data_dir = os.path.join(project_root, "data_folder")

    # helper to read CSV with CSVHandler
    def read_csv_file(fname):
        path = os.path.join(data_dir, fname)
        df = CSVHandler(path).read()
        if df is None:
            raise FileNotFoundError(f"Failed to read {path}")
        return df

    # 1) load demand
    demand_df = read_csv_file("demand.csv")

    # 2) load CA_holding.csv for holding costs (warehouse & store)
    h = {}
    df_h = read_csv_file("CA_holding.csv")
    # df_h columns: [store_id, item_id, wh_holding, st_holding]
    # Map warehouse (n_code=1) and store (n_code=2)
    for row in df_h.itertuples(index=False):
        item = row.item_id
        wh_cost = float(row.wh_holding)
        st_cost = float(row.st_holding)
        h[(item, 1)] = wh_cost
        h[(item, 2)] = st_cost

    # 3) load distance
    dist_df = read_csv_file("distance.csv")

    # 4) load all items
    item_df = read_csv_file("item.csv")

    # 5) optional beta
    beta_path = os.path.join(data_dir, "beta.csv")
    if os.path.exists(beta_path):
        beta = float(CSVHandler(beta_path).read()["beta"].iloc[0])
    else:
        beta = 0.1

    # Build other param dicts
    D = {(row.i, row.n, int(row.t)): float(row.D) for row in demand_df.itertuples()}
    d = {(row.n_from, row.n_to): float(row.d) for row in dist_df.itertuples()}

    # Define sets
    T = sorted(demand_df["t"].astype(int).unique())
    # P: perishable items (those present in h keys)
    P = sorted({i for (i, n) in h.keys()})
    # NP: non-perishable items from item_df excluding P
    all_items = sorted(item_df["i"].unique())
    NP = [i for i in all_items if i not in P]

    all_arcs = list(d.keys())
    ws_arcs = [(u, v) for (u, v) in all_arcs if u.startswith("W") and v.startswith("S")]

    # default max_perishable_age from env or 2
    max_age_env = os.getenv("MAX_PERISHABLE_AGE")
    max_perishable_age = (
        int(max_age_env) if max_age_env and max_age_env.isdigit() else 2
    )
    A_P = list(range(1, max_perishable_age + 1))

    sets = {
        "T": T,
        "P": P,
        "NP": NP,
        "A_P": A_P,
        "all_arcs": all_arcs,
        "nodes_F": [1],
        "nodes_WS": [1],
        "nodes_S": [1, 2, 3],
    }

    params = {
        "D": D,
        "h": h,
        "U": 500.0,
        "C": 100.0,
        "d": d,
        "alpha": 5,
        "beta": beta,
        "max_perishable_age": max_perishable_age,
    }

    print(f"[INFO] Loaded data from {data_dir} (max_age={max_perishable_age})")
    return sets, params
