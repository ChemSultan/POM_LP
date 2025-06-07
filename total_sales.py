import os
import pandas as pd

# 1) 폴더 및 파일명 목록 설정
base_path = "test_data/demand_forecast/jw_prophet"
files = [
    "jw_CA_1.csv",
    "jw_CA_2.csv",
    "jw_CA_3.csv",
    "jw_CA_4.csv",
    "jw_TX_1.csv",
    "jw_TX_2.csv",
    "jw_TX_3.csv",
    "jw_WI_1.csv",
    "jw_WI_2.csv",
    "jw_WI_3.csv",
]

# 2) 각 파일별 합계 및 전체 합계 초기화
total_sales = 0.0
file_sums = {}

# 3) 파일 순회하며 읽고, 'item_name' 열 제외한 나머지(주별 수요) 합산
for fname in files:
    path = os.path.join(base_path, fname)
    df = pd.read_csv(path)
    # 'item_name' 컬럼을 제외한 모든 주별 수요 값을 합산
    sum_val = df.drop(columns=["item_name"]).values.sum()
    file_sums[fname] = sum_val
    total_sales += sum_val

# 4) 결과를 DataFrame으로 정리
results_df = pd.DataFrame(
    [{"file": fname, "sales_sum": file_sums[fname]} for fname in files]
)
results_df.loc[len(results_df)] = ["Total", total_sales]

# 5) 출력
print(results_df)
