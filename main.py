from handler.data_handler import load_data
from lp_solver import LPSolver

# 1) 데이터 로드
sets, params = load_data("data")

# 2) 모델 생성 및 최적화
solver = LPSolver(sets, params)
solver.solve()
print("hi")
