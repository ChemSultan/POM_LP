import gurobipy as gp
from gurobipy import GRB

# 모델 생성
model = gp.Model("simple_lp")

# 변수 추가
x = model.addVar(lb=0, name="x")
y = model.addVar(lb=0, name="y")

# 목적함수 설정
model.setObjective(x + y, GRB.MAXIMIZE)

# 제약조건 추가
model.addConstr(x + 2 * y <= 4, "c1")
model.addConstr(4 * x + 2 * y <= 12, "c2")

# 최적화 수행
model.optimize()

# 결과 출력
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal}")
    for v in model.getVars():
        print(f"{v.varName} = {v.x}")
else:
    print("No optimal solution found.")
