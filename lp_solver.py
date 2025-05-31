import math
import gurobipy as gp
from gurobipy import GRB

from itertools import product


class LPSolver:
    def __init__(self, sets: dict, params: dict):
        """
        sets: dict with the following keys:
            - 'T': list of time periods
            - 'P': list of perishable items
            - 'NP': list of non-perishable items
            - 'A_P': list of age indices for perishables
            - 'all_arcs': list of tuples (n_from, n_to) for all transport arcs (F->W and W->S)
            - 'nodes_F': list of factory nodes
            - 'nodes_WS': list of warehouse + store nodes
            - 'nodes_S': list of store nodes
        params: dict with the following keys:
            - 'D': dict mapping (i, n, t) -> demand
            - 'h': dict mapping (i, n) -> holding cost
            - 'U': float, capacity per transport unit
            - 'C': dict mapping n -> inventory capacity
            - 'd': dict mapping (n_from, n_to) -> distance
            - 'alpha': dict mapping i -> transport cost coeff
            - 'beta': float, variable transport cost coeff
        """
        self.sets = sets
        self.params = params
        self.model = gp.Model("InventoryTransportModel")
        self._build_vars()
        self._build_objective()
        self._build_constraints()

    def _build_vars(self):
        s = self.sets
        # decision variables (all continuous)
        self.x = self.model.addVars(
            s["NP"] + s["P"],
            s["all_arcs"],
            s["T"],
            name="x",
            vtype=GRB.CONTINUOUS,
            lb=0.0,
        )
        self.N = self.model.addVars(
            s["all_arcs"], s["T"], name="N", vtype=GRB.CONTINUOUS, lb=0.0
        )
        self.I_P = self.model.addVars(
            s["P"],
            s["nodes_WS"],
            s["T"],
            s["A_P"],
            name="I_P",
            vtype=GRB.CONTINUOUS,
            lb=0.0,
        )
        self.I_NP = self.model.addVars(
            s["NP"], s["nodes_S"], s["T"], name="I_NP", vtype=GRB.CONTINUOUS, lb=0.0
        )
        self.s_P = self.model.addVars(
            s["P"],
            s["nodes_WS"],
            s["T"],
            s["A_P"],
            name="s_P",
            vtype=GRB.CONTINUOUS,
            lb=0.0,
        )
        self.s_NP = self.model.addVars(
            s["NP"], s["nodes_S"], s["T"], name="s_NP", vtype=GRB.CONTINUOUS, lb=0.0
        )

    def _build_objective(self):
        s = self.sets
        p = self.params
        m = self.model
        # Holding costs
        expr = gp.quicksum(
            p["h"][(i, n)] * self.I_NP[i, n, t]
            for i in s["NP"]
            for n in s["nodes_S"]
            for t in s["T"]
        )
        expr += gp.quicksum(
            p["h"][(i, n)] * self.I_P[i, n, t, a]
            for i in s["P"]
            for n in s["nodes_WS"]
            for t in s["T"]
            for a in s["A_P"]
        )
        # Transportation costs
        expr += gp.quicksum(
            p["alpha"][i] * p["d"][(n_from, n_to)] * self.x[i, (n_from, n_to), t]
            + p["beta"] * p["d"][(n_from, n_to)] * self.N[(n_from, n_to), t]
            for i in s["NP"] + s["P"]
            for (n_from, n_to) in s["all_arcs"]
            for t in s["T"]
        )
        m.setObjective(expr, GRB.MINIMIZE)

    def _build_constraints(self):
        s = self.sets
        p = self.params
        m = self.model
        # (T1) Transportation capacity on all arcs
        for n_from, n_to in s["all_arcs"]:
            for t in s["T"]:
                m.addConstr(
                    gp.quicksum(self.x[i, (n_from, n_to), t] for i in s["NP"] + s["P"])
                    <= p["U"] * self.N[(n_from, n_to), t],
                    name=f"T1_{n_from}_{n_to}_{t}",
                )
        # (P1) Perishable: Initial inventory at stores
        for i in s["P"]:
            for n in s["nodes_S"]:
                max_age = max(s["A_P"])
                m.addConstr(
                    self.I_P[i, n, 0, max_age] == math.ceil(p["D"][(i, n, 1)]),
                    name=f"P1_{i}_{n}",
                )
        # (P2) Perishable: New stock arrival on all arcs
        for i in s["P"]:
            for n_from, n_to in s["all_arcs"]:
                for t in s["T"]:
                    m.addConstr(
                        self.I_P[i, n_to, t, 1] == self.x[i, (n_from, n_to), t],
                        name=f"P2_{i}_{n_to}_{t}",
                    )
        # (P3) Perishable: Aging & sales flow
        for i in s["P"]:
            for n in s["nodes_WS"]:
                for t in s["T"][:-1]:
                    for a in s["A_P"][:-1]:
                        m.addConstr(
                            self.I_P[i, n, t + 1, a + 1]
                            == self.I_P[i, n, t, a] - self.s_P[i, n, t, a],
                            name=f"P3_{i}_{n}_{t}_{a}",
                        )
        # (P4) Perishable: Demand fulfillment at stores
        for i in s["P"]:
            for n in s["nodes_S"]:
                for t in s["T"]:
                    m.addConstr(
                        gp.quicksum(self.s_P[i, n, t, a] for a in s["A_P"])
                        >= p["D"][(i, n, t)],
                        name=f"P4_{i}_{n}_{t}",
                    )
        # (P5) Perishable: Sales capacity
        for i in s["P"]:
            for n in s["nodes_S"]:
                for t in s["T"]:
                    for a in s["A_P"]:
                        m.addConstr(
                            self.s_P[i, n, t, a] <= self.I_P[i, n, t, a],
                            name=f"P5_{i}_{n}_{t}_{a}",
                        )
        # (P6) Perishable: Inventory capacity
        for n in s["nodes_WS"]:
            for t in s["T"]:
                m.addConstr(
                    gp.quicksum(self.I_P[i, n, t, a] for i in s["P"] for a in s["A_P"])
                    <= p["C"][n],
                    name=f"P6_{n}_{t}",
                )
        # (NP1) Non-perishable: Initial inventory at stores
        for i in s["NP"]:
            for n in s["nodes_S"]:
                m.addConstr(
                    self.I_NP[i, n, 0] == math.ceil(p["D"][(i, n, 1)]),
                    name=f"NP1_{i}_{n}",
                )
        # (NP2) Non-perishable: Inventory flow on all arcs
        for i in s["NP"]:
            for n_from, n_to in s["all_arcs"]:
                for t in s["T"][:-1]:
                    m.addConstr(
                        self.I_NP[i, n_to, t + 1]
                        == self.I_NP[i, n_to, t]
                        + self.x[i, (n_from, n_to), t]
                        - self.s_NП[i, n_to, t],
                        name=f"NP2_{i}_{n_to}_{t}",
                    )
        # (NP3) Non-perishable: Demand fulfillment
        for i in s["NP"]:
            for n in s["nodes_S"]:
                for t in s["T"]:
                    m.addConstr(
                        self.s_NP[i, n, t] >= p["D"][(i, n, t)], name=f"NP3_{i}_{n}_{t}"
                    )
        # (NP4) Non-perishable: Inventory capacity
        for n in s["nodes_WS"]:
            for t in s["T"]:
                m.addConstr(
                    gp.quicksum(self.I_NP[i, n, t] for i in s["NP"]) <= p["C"][n],
                    name=f"NP4_{n}_{t}",
                )

    def solve(self):
        """Optimize the model and print results"""
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(f"[RESULT] Objective: {self.model.objVal}")
            for v in self.model.getVars():
                print(f"{v.varName} = {v.x}")
        else:
            print("[ERROR] No optimal solution found.")
