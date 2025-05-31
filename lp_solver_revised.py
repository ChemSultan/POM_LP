import os
import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from handler.csv_handler import CSVHandler


class LPSolver:
    def __init__(self, data_folder: str):
        self.data_dir = os.path.join(os.getcwd(), data_folder)
        self._load_data()
        self.model = gp.Model("InventoryTransportModel")
        self._build_vars()
        self._build_objective()
        self._build_constraints()

    def _read_csv(self, filename):
        path = os.path.join(self.data_dir, filename)
        df = CSVHandler(path).read()
        if df is None:
            raise FileNotFoundError(f"Failed to read {path}")
        return df

    def _load_data(self):
        demand_df = self._read_csv("demand.csv")
        holding_df = self._read_csv("holding.csv")
        cap_df = self._read_csv("capacity.csv")
        dist_df = self._read_csv("distance.csv")
        alpha_df = self._read_csv("alpha.csv")

        beta_path = os.path.join(self.data_dir, "beta.csv")
        if os.path.exists(beta_path):
            self.beta = float(CSVHandler(beta_path).read()["beta"].iloc[0])
        else:
            self.beta = 0.1

        self.max_perishable_age = 3

        self.D = {(r.i, r.n, int(r.t)): float(r.D) for r in demand_df.itertuples()}
        self.h = {(r.i, r.n): float(r.h) for r in holding_df.itertuples()}
        self.C = {r.n: float(r.C) for r in cap_df.itertuples()}
        self.d = {(r.n_from, r.n_to): float(r.d) for r in dist_df.itertuples()}
        self.alpha = {r.i: float(r.alpha) for r in alpha_df.itertuples()}
        self.U = 100.0

        self.T = sorted(demand_df["t"].astype(int).unique())
        self.P = sorted(holding_df["i"].unique())
        self.NP = sorted(set(alpha_df["i"]) - set(self.P)) + sorted(
            set(alpha_df["i"]) & set(self.P)
        )
        self.A_P = list(range(1, self.max_perishable_age + 1))
        self.all_arcs = list(self.d.keys())
        self.ws_arcs = [
            (u, v)
            for (u, v) in self.all_arcs
            if u.startswith("W") and v.startswith("S")
        ]
        self.nodes_F = sorted(
            dist_df.loc[dist_df["n_from"].str.startswith("F"), "n_from"].unique()
        )
        self.nodes_WS = sorted(set(dist_df["n_from"]).union(dist_df["n_to"]))
        self.nodes_S = sorted(
            dist_df.loc[dist_df["n_to"].str.startswith("S"), "n_to"].unique()
        )

        print(
            f"[INFO] Loaded data from {self.data_dir} (max_age={self.max_perishable_age})"
        )
