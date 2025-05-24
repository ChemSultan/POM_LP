import os
import pandas as pd

# 1) CSV 읽기
df = pd.read_csv("sell_prices.csv")

# 2) store_id 앞 두 글자만 남기기
df["store_id"] = df["store_id"].str[:2]

# 3) store_id·item_id별 첫 번째 행만 선택
first_df = df.drop_duplicates(subset=["store_id", "item_id"], keep="first")

# 4) wm_yr_wk 컬럼 제거
first_df = first_df.drop(columns=["wm_yr_wk"])

# 5) 인덱스 리셋
first_df = first_df.reset_index(drop=True)

# 6) item_id에서 카테고리 추출 (언더바 앞부분)
first_df["category"] = first_df["item_id"].str.split("_").str[0]

# 7) 기존 holding_cost 계산
weights_orig = {"HOBBIES": 0.25, "HOUSEHOLD": 0.20, "FOODS": 0.40}
first_df["holding_cost"] = (
    first_df["sell_price"] * first_df["category"].map(weights_orig)
).round(3)

# 8) 새로운 wh_holding 계산 (소수점 셋째 자리까지)
weights_wh = {"HOBBIES": 0.125, "HOUSEHOLD": 0.10, "FOODS": 0.20}
first_df["wh_holding"] = (
    first_df["sell_price"] * first_df["category"].map(weights_wh)
).round(3)

# 9) 컬럼 재정렬 및 st_holding으로 이름 변경
first_df = first_df.rename(columns={"holding_cost": "st_holding"})
result_df = first_df[["store_id", "item_id", "wh_holding", "st_holding"]]

# 10) data_folder에 store_id별로 개별 CSV로 저장
output_dir = "data_folder"
os.makedirs(output_dir, exist_ok=True)

for store in result_df["store_id"].unique():
    store_df = result_df[result_df["store_id"] == store]
    filename = f"{store}_holding.csv"
    output_path = os.path.join(output_dir, filename)
    store_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved {store_df.shape[0]} rows to {output_path}")
