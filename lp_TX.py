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

        # ─── 3) 노드·아이템별 보관비 불러오기 ───
        self._load_specific_holding_costs()

        # ─── 4) 품목별 부피 정보 불러오기 (transportation_capacity.csv 사용) ───
        self._load_volume_info()

        # ─── 5) 거리 정보 불러오기 (distance.csv 사용) ───
        self._load_distance_info()

        # ─── 6) 카테고리별 운송비 계수 불러오기 (operation_cost_by_category.csv 사용) ───
        self._load_category_costs()

        # ─── 7) 운송 용량 상수 정의 ───
        #    store-origin일 때, U_store; warehouse-origin일 때, U_wh
        self.U_store = 7237103.422
        self.U_wh = 50659723.95

        # ─── 8) 기타 파라미터 정의 ───
        self.max_perishable_age = 3

        # ─── 9) 집합(sets) 정의 ───
        self._define_sets()

        # ─── 10) Gurobi 모델 생성 및 빌드 ───
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
        self.prophet_TX_1_df = self._read_csv("jw_TX_1.csv")
        self.prophet_TX_2_df = self._read_csv("jw_TX_2.csv")
        self.prophet_TX_3_df = self._read_csv("jw_TX_3.csv")

    def _combine_forecast(self):
        """
        prophet_CA_1_df ~ prophet_CA_4_df를 하나의 MultiIndex 구조로 합쳐서 self.D 생성
        index: (item_name, n, t), value: demand
        """
        dfs_and_nodes = [
            (self.prophet_TX_1_df, "TX_1"),
            (self.prophet_TX_2_df, "TX_2"),
            (self.prophet_TX_3_df, "TX_3"),
        ]

        long_frames = []
        for df, node_name in dfs_and_nodes:
            melted = df.melt(
                id_vars="item_name",
                var_name="t",  # "w_1", "w_2", …, "w_52"
                value_name="demand",
            )
            melted["n"] = node_name
            long_frames.append(melted)

        combined = pd.concat(long_frames, ignore_index=True)
        combined.set_index(["item_name", "n", "t"], inplace=True)
        self.D = combined["demand"]  # MultiIndex Series

    def _load_specific_holding_costs(self):
        """
        specific_CA_holding.csv, specific_TX_holding.csv, specific_WI_holding.csv를 읽어서
        (item_id, store_id) → holding cost 값을 self.holding_costs 딕셔너리로 저장.
        """
        file_list = [
            "specific_CA_holding.csv",
            "specific_TX_holding.csv",
            "specific_WI_holding.csv",
        ]
        self.holding_costs = {}  # {(item_id, store_id): cost}

        for fname in file_list:
            path = os.path.join(self.data_dir, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{fname} 파일을 찾을 수 없습니다: {path}")

            tmp = pd.read_csv(path)
            # CSV: store_id, item_id, 2017(holding cost) 컬럼을 가정
            for _, row in tmp.iterrows():
                store = row["store_id"]
                item = row["item_id"]
                cost = row["2017"]
                self.holding_costs[(item, store)] = float(cost)

    def _load_volume_info(self):
        """
        transportation_capacity.csv 를 읽어서,
        - 각 CATEGORY 당 VOLUMEPERITEM,
        - 전체 VOLUME 한계 (예: 42000)
        을 self.volume_per_item, self.volume_capacity 에 저장.
        """
        path = os.path.join(self.data_dir, "transportation_capacity.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"transportation_capacity.csv 파일을 찾을 수 없습니다: {path}"
            )

        df_vol = pd.read_csv(path)
        # 기대형태: CATEGORY, VOLUMEPERITEM, VOLUME
        # 예시:
        #   CATEGORY,VOLUMEPERITEM,VOLUME
        #   FOODS,1,42000
        #   HOBBIES,2,42000
        #   HOUSEHOLD,6,42000

        # 1) 전체 capacity — 모든 행에서 같은 수치라고 가정
        self.volume_capacity = int(df_vol["VOLUME"].iloc[0])

        # 2) category → volume per item 매핑
        self.volume_per_item = {}
        for _, row in df_vol.iterrows():
            cat = row["CATEGORY"]
            vol_per_item = float(row["VOLUMEPERITEM"])
            self.volume_per_item[cat] = vol_per_item

    def _load_distance_info(self):
        """
        distance.csv 를 읽어서,
        - STATE별로 FCWH, FCST, WHST 거리를 매핑하여
          self.distance_map[(n1, n2)] 형태로 저장.
        """
        df_dist = self.distance_df.copy()
        # 열: STATE, FCWH, FCST, WHST
        # 예: CA,374,411,177
        self.distance_map = {}

        for _, row in df_dist.iterrows():
            state = row["STATE"]
            d_fcwh = float(row["FCWH"])
            d_fcst = float(row["FCST"])
            d_whst = float(row["WHST"])

            # 노드 이름: “STATE_fac”, “STATE_wh”, “STATE_i” (i in {1,2,3,4})
            fac_node = f"{state}_fac"
            wh_node = f"{state}_wh"
            st_nodes = [f"{state}_1", f"{state}_2", f"{state}_3", f"{state}_4"]

            # (Fac, Wh)
            self.distance_map[(fac_node, wh_node)] = d_fcwh
            self.distance_map[(wh_node, fac_node)] = d_fcwh

            # (Fac, St_i) 및 (St_i, Fac)
            for st in st_nodes:
                self.distance_map[(fac_node, st)] = d_fcst
                self.distance_map[(st, fac_node)] = d_fcst

            # (Wh, St_i) 및 (St_i, Wh)
            for st in st_nodes:
                self.distance_map[(wh_node, st)] = d_whst
                self.distance_map[(st, wh_node)] = d_whst

    def _load_category_costs(self):
        """
        operation_cost_by_category.csv 를 읽어서,
        - 각 CATEGORY별 COEFFICIENT(α)와 INTERCEPT(β)을
          self.category_costs[category] = (coef, intercept) 형태로 저장.
        """
        path = os.path.join(self.data_dir, "operation_cost_by_category.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"operation_cost_by_category.csv 파일을 찾을 수 없습니다: {path}"
            )

        df_op = pd.read_csv(path)
        # 형태: CATEGORY, COEFFICIENT, INTERCEPT
        # 예:
        #   FOODS,3.01122E-07,1.021075236
        #   HOBBIES,6.02244E-07,1.021075236
        #   HOUSEHOLDS,1.80673E-06,1.021075236

        self.category_costs = {}
        for _, row in df_op.iterrows():
            cat = row["CATEGORY"]
            coef = float(row["COEFFICIENT"])
            intercept = float(row["INTERCEPT"])
            self.category_costs[cat] = (coef, intercept)

    def _define_sets(self):
        """
        - 아이템 집합 P, NP, All 정의
        - 노드 집합 Fac, Wh, St 정의
        - 전체 노드(AllNodes) = Fac ∪ Wh ∪ St
        - 기간(periods), 연령(ages) 정의
        """
        all_items = list(self.D.index.get_level_values("item_name").unique())

        # 부패성 vs 비부패성 구분
        self.P = [i for i in all_items if i.startswith("FOODS")]
        self.NP = [i for i in all_items if i.startswith("HOUSEHOLD")]
        self.AllItems = all_items

        # 노드 집합 정의
        self.Fac = ["TX_fac"]
        self.Wh = ["TX_wh"]
        self.St = ["TX_1", "TX_2", "TX_3"]

        self.AllNodes = self.Fac + self.Wh + self.St

        # 기간(periods): D의 t 레벨
        unique_periods = list(self.D.index.get_level_values("t").unique())
        self.periods = sorted(unique_periods, key=lambda x: int(x.split("_")[1]))

        # 연령(ages): 1 ~ max_perishable_age
        self.ages = list(range(1, self.max_perishable_age + 1))

    def _build_vars(self):
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
            self.P,
            self.AllNodes,
            periods,
            self.ages,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="I_P",
        )
        # (4) I_NP[i, n, t]
        self.I_NP = self.model.addVars(
            self.NP, self.AllNodes, periods, vtype=GRB.CONTINUOUS, lb=0.0, name="I_NP"
        )
        # (5) s_P[i, n, t, a]
        self.s_P = self.model.addVars(
            self.P,
            self.AllNodes,
            periods,
            self.ages,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="s_P",
        )
        # (6) s_NP[i, n, t]
        self.s_NP = self.model.addVars(
            self.NP, self.AllNodes, periods, vtype=GRB.CONTINUOUS, lb=0.0, name="s_NP"
        )
        self.model.update()

    def _build_objective(self):
        """
        Objective:
          ∑_{t}
            [ ∑_{i∈NP} ∑_{n∈AllNodes} h_{i,n} · I_NP[i,n,t]
            + ∑_{i∈P} ∑_{n∈AllNodes} ∑_{a∈ages} h_{i,n} · I_P[i,n,t,a]
            + ∑_{i∈AllItems} ∑_{(n₁,n₂)∈(Fac×Wh) ∪ (Wh×St) ∪ (Fac×St)}
                ( coef_i·distance[n₁,n₂]·x[i,n₁,n₂,t]
                + intercept_i·distance[n₁,n₂]·N[n₁,n₂,t] ) ]
        """
        # (1) 아크(arcs) 생성
        arcs = []
        for f in self.Fac:
            for w in self.Wh:
                arcs.append((f, w))
        for w in self.Wh:
            for s in self.St:
                arcs.append((w, s))
        for f in self.Fac:
            for s in self.St:
                arcs.append((f, s))

        obj_expr = gp.LinExpr()

        # (2-1) 비-부패성 보관 비용
        for i in self.NP:
            for n in self.AllNodes:
                for t in self.periods:
                    h_in = self.holding_costs.get((i, n), 0.0)
                    obj_expr.add(h_in * self.I_NP[i, n, t])

        # (2-2) 부패성 보관 비용
        for i in self.P:
            for n in self.AllNodes:
                for t in self.periods:
                    for a in self.ages:
                        h_in = self.holding_costs.get((i, n), 0.0)
                        obj_expr.add(h_in * self.I_P[i, n, t, a])

        # (2-3) 운송 비용 항목: coef_i·distance·x + intercept_i·distance·N
        for i in self.AllItems:
            category = i.split("_")[0]
            coef_i, intercept_i = self.category_costs.get(category, (0.0, 0.0))
            for n1, n2 in arcs:
                dist = self.distance_map.get((n1, n2), 0.0)
                for t in self.periods:
                    obj_expr.add(coef_i * dist * self.x[i, n1, n2, t])
                    obj_expr.add(intercept_i * dist * self.N[n1, n2, t])

        # (3) 모델에 목적함수 설정 (Minimize)
        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def _build_constraints(self):
        """
        제약조건 구현:
        (T1) Transportation capacity  → 부피 및 U_store/U_wh 반영
        (P1)~(P6) Perishable inventory constraints (부피 기반)
        (NP1)~(NP4) Non-perishable inventory constraints (부피 기반)
        """
        P_items = self.P
        NP_items = self.NP
        all_items = self.AllItems

        Fac = self.Fac
        Wh = self.Wh
        St = self.St
        periods = self.periods
        ages = self.ages

        # (A) 아크(arcs) 생성
        arcs = []
        for f in Fac:
            for w in Wh:
                arcs.append((f, w))
        for w in Wh:
            for s in St:
                arcs.append((w, s))

        first_period = periods[0]
        max_age = max(ages)

        # (T1) Transportation capacity  ⟶ “부피 합 ≤ U_origin × N”
        #   U_origin = U_store if origin is a store(“*_1” etc)
        #            = U_wh    if origin is warehouse(“*_wh”)
        #            = U_wh    (기타, 예: facility도 U_wh로 처리)
        for n1, n2 in arcs:
            for t in periods:
                expr = gp.LinExpr()
                for i in all_items:
                    category = i.split("_")[0]
                    vpi = self.volume_per_item.get(category, 0.0)
                    expr.add(vpi * self.x[i, n1, n2, t])
                # origin n1별 U 선택
                if n1.endswith("_wh"):
                    U_origin = self.U_wh
                elif any(n1.endswith(f"_{i}") for i in ["1", "2", "3", "4"]):
                    U_origin = self.U_store
                else:
                    # facility(예: “CA_fac”)도 warehouse 수준으로 간주
                    U_origin = self.U_wh
                self.model.addConstr(
                    expr <= U_origin * self.N[n1, n2, t], name=f"T1_vol_U_{n1}_{n2}_{t}"
                )

        # (P1) Perishable: Initial inventory
        for i in P_items:
            for n in St:
                demand_val = self.D[i, n, first_period]
                init_inv = math.ceil(demand_val)
                self.model.addConstr(
                    self.I_P[i, n, first_period, max_age] == init_inv,
                    name=f"P1_init_{i}_{n}_{first_period}",
                )

        # (P2) Perishable: New stock arrival
        for i in P_items:
            for n1, n2 in arcs:
                for t in periods:
                    self.model.addConstr(
                        self.I_P[i, n2, t, 1] == self.x[i, n1, n2, t],
                        name=f"P2_new_{i}_{n1}_{n2}_{t}",
                    )

        # (P3) Perishable: Aging & sales flow
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

        # (P4) Perishable: Demand fulfillment
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

        # (P5) Perishable: Sales capacity
        for i in P_items:
            for n in St:
                for t in periods:
                    for a in ages:
                        self.model.addConstr(
                            self.s_P[i, n, t, a] <= self.I_P[i, n, t, a],
                            name=f"P5_salescap_{i}_{n}_{t}_a{a}",
                        )

        # (P6) Perishable: Inventory capacity ─── 부피 단위로 제한
        for n in Wh + St:
            for t in periods:
                expr = gp.LinExpr()
                for i in P_items:
                    for a in ages:
                        category = i.split("_")[0]
                        vpi = self.volume_per_item.get(category, 0.0)
                        expr.add(vpi * self.I_P[i, n, t, a])
                self.model.addConstr(
                    expr <= self.volume_capacity, name=f"P6_invcap_volume_{n}_{t}"
                )

        # (NP1) Non-perishable: Initial inventory
        for i in NP_items:
            for n in St:
                demand_val = self.D[i, n, first_period]
                init_inv = math.ceil(demand_val)
                self.model.addConstr(
                    self.I_NP[i, n, first_period] == init_inv,
                    name=f"NP1_init_{i}_{n}_{first_period}",
                )

        # (NP2) Non-perishable: Inventory flow
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

        # (NP3) Non-perishable: Demand fulfillment
        for i in NP_items:
            for n in St:
                for t in periods:
                    demand_val = self.D[i, n, t]
                    self.model.addConstr(
                        self.s_NP[i, n, t] >= demand_val, name=f"NP3_demand_{i}_{n}_{t}"
                    )

        # (NP4) Non-perishable: Inventory capacity ─── 부피 단위로 제한
        for n in Wh + St:
            for t in periods:
                expr = gp.LinExpr()
                for i in NP_items:
                    category = i.split("_")[0]
                    vpi = self.volume_per_item.get(category, 0.0)
                    expr.add(vpi * self.I_NP[i, n, t])
                self.model.addConstr(
                    expr <= self.volume_capacity, name=f"NP4_invcap_volume_{n}_{t}"
                )


if __name__ == "__main__":
    # 1) Solver 객체 생성: 데이터 로딩 및 모델 구축까지 자동 수행
    solver = LPSolver(data_folder="test_data/data_folder")

    # 2) 최적화 실행
    solver.model.optimize()

    # 3) 최적화 결과 확인 (var.Obj * var.X 방식)
    if solver.model.Status == GRB.OPTIMAL:
        holding_np = sum(var.Obj * var.X for var in solver.I_NP.values())
        holding_p = sum(var.Obj * var.X for var in solver.I_P.values())
        trans_x = sum(var.Obj * var.X for var in solver.x.values())
        trans_N = sum(var.Obj * var.X for var in solver.N.values())
        total = holding_np + holding_p + trans_x + trans_N

        print("\n▶ TX 최적화 완료\n")
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
