import os
import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from handler.csv_handler import CSVHandler


class LPSolver:
    def __init__(self, data_folder: str):
        # ─── 1) 데이터 폴더 경로 설정 ───
        self.data_dir = os.path.join(os.getcwd(), data_folder)

        # ─── 2) CSV 불러오기 + 예측 데이터 통합 ───
        self._load_data()  # CSV 파일 읽어와서 데이터프레임에 저장
        self._combine_forecast()  # prophet_CA_i_df 네 개를 합쳐서 self.D 생성

        # ─── 3) 파라미터 정의 (max_perishable_age 먼저) ───
        self.max_perishable_age = 3
        self.h = 1
        self.U = 10000
        self.C = 30000
        self.d = 200
        self.alpha = 6.02244e-07
        self.beta = 1.021075236

        # ─── 4) 집합(sets) 정의 ───
        self._define_sets()

        # ─── 5) Gurobi 모델 생성 및 빌드 ───
        self.model = gp.Model("InventoryTransportModel")
        self._build_vars()
        self._build_objective()
        self._build_constraints()

    def _read_csv(self, filename):
        """
        filename에 따라 test_data/data_folder 또는
        test_data/demand_forecast/jw_prophet 에서 파일을 읽어옵니다.
        """
        if filename in [
            "capacity_warehouse.csv",
            "transportation_capacity.csv",
            "holding_cost.csv",
            "operation_cost_by_category.csv",
            "distance.csv",
        ]:
            path = os.path.join("test_data", "data_folder", filename)
        else:
            path = os.path.join("test_data", "demand_forecast", "jw_prophet", filename)
        return pd.read_csv(path)

    def _load_data(self):
        # ───── 기본 데이터 불러오기 ─────
        self.node_capa_df = self._read_csv("capacity_warehouse.csv")
        self.transport_capa_df = self._read_csv("transportation_capacity.csv")
        self.operation_cost_df = self._read_csv("operation_cost_by_category.csv")
        self.distance_df = self._read_csv("distance.csv")

        # ───── 예측 데이터(네 개) 불러오기 ─────
        self.prophet_CA_1_df = self._read_csv("jw_CA_1.csv")
        self.prophet_CA_2_df = self._read_csv("jw_CA_2.csv")
        self.prophet_CA_3_df = self._read_csv("jw_CA_3.csv")
        self.prophet_CA_4_df = self._read_csv("jw_CA_4.csv")

    def _combine_forecast(self):
        """
        prophet_CA_1_df ~ prophet_CA_4_df를 하나의 MultiIndex 구조로 합쳐서 self.D 생성
        index: (item_name, n, t), value: demand
        """
        dfs_and_nodes = [
            (self.prophet_CA_1_df, "CA_1"),
            (self.prophet_CA_2_df, "CA_2"),
            (self.prophet_CA_3_df, "CA_3"),
            (self.prophet_CA_4_df, "CA_4"),
        ]

        long_frames = []
        for df, node_name in dfs_and_nodes:
            melted = df.melt(
                id_vars="item_name",
                var_name="t",  # "w_1", "w_2", …, "w_52"
                value_name="demand",  # 예측 수요량
            )
            melted["n"] = node_name
            long_frames.append(melted)

        combined = pd.concat(long_frames, ignore_index=True)
        combined.set_index(["item_name", "n", "t"], inplace=True)
        self.D = combined["demand"]  # MultiIndex Series

    def _define_sets(self):
        """
        - 아이템 집합 P, NP, All 정의
        - 노드 집합 Fac, Wh, St 정의
        - 전체 노드(AllNodes) = Fac ∪ Wh ∪ St
        - 기간(periods), 연령(ages) 정의
        """

        # 1) 전체 아이템 = D의 첫 번째 인덱스 레벨(item_name)
        all_items = list(self.D.index.get_level_values("item_name").unique())

        # 2) 부패성 vs 비부패성 구분
        self.P = [i for i in all_items if i.startswith("FOODS")]
        self.NP = [i for i in all_items if i.startswith("HOUSEHOLD")]
        self.AllItems = all_items  # P ∪ NP

        # 3) 노드 집합 정의
        self.Fac = ["CA_fac"]  # \mathcal{F}
        self.Wh = ["CA_wh"]  # \mathcal{W}
        self.St = ["CA_1", "CA_2", "CA_3", "CA_4"]  # \mathcal{S}

        # 4) 전체 노드 집합
        self.AllNodes = self.Fac + self.Wh + self.St

        # 5) 기간(periods): D의 두 번째 인덱스 레벨("t")
        unique_periods = list(self.D.index.get_level_values("t").unique())
        self.periods = sorted(unique_periods, key=lambda x: int(x.split("_")[1]))

        # 6) 연령(ages): 1 ~ max_perishable_age
        self.ages = list(range(1, self.max_perishable_age + 1))

    def _build_vars(self):
        """
        변수 정의 단계에서는 이미 _define_sets()에서:
          - self.P, self.NP, self.AllItems
          - self.Fac, self.Wh, self.St, self.AllNodes
          - self.periods, self.ages
        등이 생성되어 있기 때문에, _build_vars() 안에서는 집합만 참조해서 변수 추가하면 됩니다.
        """

        items = self.AllItems
        nodes = self.AllNodes
        periods = self.periods
        ages = self.ages

        # (1) x[i, n_origin, n_dest, t]
        self.x = self.model.addVars(
            items, nodes, nodes, periods, vtype=GRB.CONTINUOUS, lb=0.0, name="x"
        )
        # (2) N[n_origin, n_dest, t]
        self.N = self.model.addVars(
            nodes, nodes, periods, vtype=GRB.CONTINUOUS, lb=0.0, name="N"
        )
        # (3) I_P[i, n, t, a]
        self.I_P = self.model.addVars(
            self.P, nodes, periods, ages, vtype=GRB.CONTINUOUS, lb=0.0, name="I_P"
        )
        # (4) I_NP[i, n, t]
        self.I_NP = self.model.addVars(
            self.NP, nodes, periods, vtype=GRB.CONTINUOUS, lb=0.0, name="I_NP"
        )
        # (5) s_P[i, n, t, a]
        self.s_P = self.model.addVars(
            self.P, nodes, periods, ages, vtype=GRB.CONTINUOUS, lb=0.0, name="s_P"
        )
        # (6) s_NP[i, n, t]
        self.s_NP = self.model.addVars(
            self.NP, nodes, periods, vtype=GRB.CONTINUOUS, lb=0.0, name="s_NP"
        )

        self.model.update()

        # (기존의 디버그 출력 부분 모두 제거)

    def _build_objective(self):
        """
        Objective:
          ∑_{t}
            [ ∑_{i∈NP} ∑_{n∈AllNodes} h · I_NP[i,n,t]
            + ∑_{i∈P} ∑_{n∈AllNodes} ∑_{a∈ages} h · I_P[i,n,t,a]
            + ∑_{i∈AllItems} ∑_{(n₁,n₂)∈(Fac×Wh) ∪ (Wh×St) ∪ (Fac×St)}
                (α·d·x[i,n₁,n₂,t] + β·d·N[n₁,n₂,t]) ]
        """

        # 1) 아크(arcs) 생성: (Fac×Wh) ∪ (Wh×St) ∪ (Fac×St)
        arcs = []
        # Fac × Wh
        for f in self.Fac:
            for w in self.Wh:
                arcs.append((f, w))
        # Wh × St
        for w in self.Wh:
            for s in self.St:
                arcs.append((w, s))
        # Fac × St
        for f in self.Fac:
            for s in self.St:
                arcs.append((f, s))

        # 2) 목적식 표현식 초기화
        obj_expr = gp.LinExpr()

        # 2-1) 비-부패성 보관 비용
        for i in self.NP:
            for n in self.AllNodes:
                for t in self.periods:
                    obj_expr.add(self.h * self.I_NP[i, n, t])

        # 2-2) 부패성 보관 비용
        for i in self.P:
            for n in self.AllNodes:
                for t in self.periods:
                    for a in self.ages:
                        obj_expr.add(self.h * self.I_P[i, n, t, a])

        # 2-3) 운송 비용 항목
        for i in self.AllItems:
            for n1, n2 in arcs:
                for t in self.periods:
                    obj_expr.add(self.alpha * self.d * self.x[i, n1, n2, t])
                    obj_expr.add(self.beta * self.d * self.N[n1, n2, t])

        # 3) 모델에 목적함수 설정 (Minimize)
        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def _build_constraints(self):
        """
        제약조건 구현:

        (T1) Transportation capacity:
          ∑_{i∈P∪NP} x[i, n', n, t] ≤ U · N[n', n, t]
          ∀ (n', n) ∈ (Fac×Wh) ∪ (Wh×St), ∀ t ∈ periods

        (P1) Perishable: Initial inventory
          I_P[i, n, first_period, max_age] = ceil(D[i, n, first_period])
          ∀ i ∈ P, n ∈ St

        (P2) Perishable: New stock arrival
          I_P[i, n, t, 1] = x[i, n', n, t]
          ∀ i ∈ P, (n', n) ∈ (Fac×Wh) ∪ (Wh×St), ∀ t ∈ periods

        (P3) Perishable: Aging & sales flow
          I_P[i, n, t_next, a+1] = I_P[i, n, t, a] - s_P[i, n, t, a]
          ∀ i ∈ P, n ∈ Wh ∪ St, ∀ t = periods[0]…periods[-2], ∀ a = 1…max_age-1

        (P4) Perishable: Demand fulfillment
          ∑_{a∈ages} s_P[i, n, t, a] ≥ D[i, n, t]
          ∀ i ∈ P, n ∈ St, ∀ t ∈ periods

        (P5) Perishable: Sales capacity
          s_P[i, n, t, a] ≤ I_P[i, n, t, a]
          ∀ i ∈ P, n ∈ St, ∀ t ∈ periods, ∀ a ∈ ages

        (P6) Perishable: Inventory capacity
          ∑_{i∈P} ∑_{a∈ages} I_P[i, n, t, a] ≤ C
          ∀ n ∈ Wh ∪ St, ∀ t ∈ periods

        (NP1) Non-perishable: Initial inventory
          I_NP[i, n, first_period] = ceil(D[i, n, first_period])
          ∀ i ∈ NP, n ∈ St

        (NP2) Non-perishable: Inventory flow
          I_NP[i, n, t_next] = I_NP[i, n, t] + x[i, n', n, t] - s_NP[i, n, t]
          ∀ i ∈ NP, (n', n) ∈ (Fac×Wh) ∪ (Wh×St), ∀ t = periods[0]…periods[-2]

        (NP3) Non-perishable: Demand fulfillment
          s_NP[i, n, t] ≥ D[i, n, t]
          ∀ i ∈ NP, n ∈ St, ∀ t ∈ periods

        (NP4) Non-perishable: Inventory capacity
          ∑_{i∈NP} I_NP[i, n, t] ≤ C
          ∀ n ∈ Wh ∪ St, ∀ t ∈ periods
        """

        # 1) 집합 참조
        P_items = self.P
        NP_items = self.NP
        all_items = self.AllItems

        Fac = self.Fac
        Wh = self.Wh
        St = self.St
        AllNodes = self.AllNodes

        periods = self.periods
        ages = self.ages

        # 2) (Fac×Wh) ∪ (Wh×St) 아크 목록 생성
        arcs = []
        for f in Fac:
            for w in Wh:
                arcs.append((f, w))
        for w in Wh:
            for s in St:
                arcs.append((w, s))

        # 3) 첫 번째 기간(first_period) 및 max_age 구하기
        first_period = periods[0]
        max_age = max(ages)

        # 4) (T1) Transportation capacity
        for n1, n2 in arcs:
            for t in periods:
                expr = gp.LinExpr()
                for i in all_items:
                    expr.add(self.x[i, n1, n2, t])
                self.model.addConstr(
                    expr <= self.U * self.N[n1, n2, t], name=f"T1_{n1}_{n2}_{t}"
                )

        # 5) (P1) Perishable: Initial inventory
        for i in P_items:
            for n in St:
                demand_val = self.D[i, n, first_period]
                init_inv = math.ceil(demand_val)
                self.model.addConstr(
                    self.I_P[i, n, first_period, max_age] == init_inv,
                    name=f"P1_init_{i}_{n}_{first_period}",
                )

        # 6) (P2) Perishable: New stock arrival
        for i in P_items:
            for n1, n2 in arcs:
                for t in periods:
                    self.model.addConstr(
                        self.I_P[i, n2, t, 1] == self.x[i, n1, n2, t],
                        name=f"P2_new_{i}_{n1}_{n2}_{t}",
                    )

        # 7) (P3) Perishable: Aging & sales flow
        for i in P_items:
            for n in Wh + St:
                for idx_t in range(len(periods) - 1):
                    t = periods[idx_t]
                    next_t = periods[idx_t + 1]
                    for a in range(1, max_age):
                        self.model.addConstr(
                            self.I_P[i, n, next_t, a + 1]
                            == self.I_P[i, n, t, a] - self.s_P[i, n, t, a],
                            name=f"P3_age_{i}_{n}_{t}_a{a}",
                        )

        # 8) (P4) Perishable: Demand fulfillment
        for i in P_items:
            for n in St:
                for t in periods:
                    expr = gp.LinExpr()
                    for a in ages:
                        expr.add(self.s_P[i, n, t, a])
                    demand_val = self.D[i, n, t]
                    self.model.addConstr(
                        expr >= demand_val, name=f"P4_demand_{i}_{n}_{t}"
                    )

        # 9) (P5) Perishable: Sales capacity
        for i in P_items:
            for n in St:
                for t in periods:
                    for a in ages:
                        self.model.addConstr(
                            self.s_P[i, n, t, a] <= self.I_P[i, n, t, a],
                            name=f"P5_salescap_{i}_{n}_{t}_a{a}",
                        )

        # 10) (P6) Perishable: Inventory capacity
        for n in Wh + St:
            for t in periods:
                expr = gp.LinExpr()
                for i in P_items:
                    for a in ages:
                        expr.add(self.I_P[i, n, t, a])
                self.model.addConstr(expr <= self.C, name=f"P6_invcap_{n}_{t}")

        # 11) (NP1) Non-perishable: Initial inventory
        for i in NP_items:
            for n in St:
                demand_val = self.D[i, n, first_period]
                init_inv = math.ceil(demand_val)
                self.model.addConstr(
                    self.I_NP[i, n, first_period] == init_inv,
                    name=f"NP1_init_{i}_{n}_{first_period}",
                )

        # 12) (NP2) Non-perishable: Inventory flow
        for i in NP_items:
            for n1, n2 in arcs:
                for idx_t in range(len(periods) - 1):
                    t = periods[idx_t]
                    next_t = periods[idx_t + 1]
                    self.model.addConstr(
                        self.I_NP[i, n2, next_t]
                        == self.I_NP[i, n2, t]
                        + self.x[i, n1, n2, t]
                        - self.s_NP[i, n2, t],
                        name=f"NP2_flow_{i}_{n1}_{n2}_{t}",
                    )

        # 13) (NP3) Non-perishable: Demand fulfillment
        for i in NP_items:
            for n in St:
                for t in periods:
                    demand_val = self.D[i, n, t]
                    self.model.addConstr(
                        self.s_NP[i, n, t] >= demand_val, name=f"NP3_demand_{i}_{n}_{t}"
                    )

        # 14) (NP4) Non-perishable: Inventory capacity
        for n in Wh + St:
            for t in periods:
                expr = gp.LinExpr()
                for i in NP_items:
                    expr.add(self.I_NP[i, n, t])
                self.model.addConstr(expr <= self.C, name=f"NP4_invcap_{n}_{t}")


if __name__ == "__main__":
    # 1) Solver 객체 생성: 데이터 로딩 및 모델 구축까지 자동 수행
    solver = LPSolver(data_folder="test_data/data_folder")

    # 2) 최적화 실행
    solver.model.optimize()

    # 3) 최적화 결과 확인 및 목적함수 각 항 계산/출력
    if solver.model.Status == GRB.OPTIMAL:
        # (1) 비-부패성 재고 보관비 (I_NP에 대한 목적계수 ⨯ X)
        holding_np = 0.0
        for var in solver.I_NP.values():
            # var.Obj는 'h' 값, var.X는 최적화된 재고량
            holding_np += var.Obj * var.X

        # (2) 부패성 재고 보관비 (I_P에 대한 목적계수 ⨯ X)
        holding_p = 0.0
        for var in solver.I_P.values():
            # var.Obj 역시 'h'
            holding_p += var.Obj * var.X

        # (3) 운송비 × x 항 (x에 대한 목적계수 ⨯ X)
        trans_x = 0.0
        for var in solver.x.values():
            # var.Obj는 α·d
            trans_x += var.Obj * var.X

        # (4) 운송비 × N 항 (N에 대한 목적계수 ⨯ X)
        trans_N = 0.0
        for var in solver.N.values():
            # var.Obj는 β·d
            trans_N += var.Obj * var.X

        # 네 개 항을 모두 합산 (이 값이 Gurobi ObjVal과 동일해야 함)
        total = holding_np + holding_p + trans_x + trans_N

        print("\n▶ 최적화 완료\n")
        print(f"1) 비-부패성 재고 보관비 총합 = {holding_np:.2f}")
        print(f"2)   부패성 재고 보관비 총합 = {holding_p:.2f}")
        print(f"3)          운송비(x 항) 총합 = {trans_x:.2f}")
        print(f"4)          운송비(N 항) 총합 = {trans_N:.2f}")
        print(f"\n→ 합산 목적함수 값     = {total:.2f}")
        print(f"→ Gurobi가 보고한 ObjVal = {solver.model.ObjVal:.2f}\n")

    else:
        print(
            "❌ 최적화가 정상적으로 종료되지 않았습니다. Status Code:",
            solver.model.Status,
        )
