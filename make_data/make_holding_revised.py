import os
import pandas as pd

# 1) 입력 파일 경로
input_path = os.path.join("real_data", "data_folder", "prediction_price_2017.csv")

# 2) CSV 읽기
df = pd.read_csv(input_path)


# 3) 카테고리 추출 함수
def extract_category(item_id: str) -> str:
    return item_id.split("_")[0]


df["category"] = df["item_id"].apply(extract_category)

# 4) 계수 정의
base_coeff = {
    "HOBBIES": 0.25 / 52,
    "HOUSEHOLD": 0.20 / 52,
    "FOODS": 0.40 / 52,
}

specific_coeff = {
    "HOBBIES": 0.125 / 52,
    "HOUSEHOLD": 0.10 / 52,
    "FOODS": 0.20 / 52,
}

# 5) 출력 디렉터리
output_dir = os.path.join("test_data", "data_folder")
os.makedirs(output_dir, exist_ok=True)


# 6) 지역별 처리 함수
def make_region_csv(region_prefix, n_stores, base_coeff, specific_coeff):
    # 6-1) 해당 지역의 base 행(예: "CA_1"~"CA_4")만 필터링
    store_ids = [f"{region_prefix}_{i}" for i in range(1, n_stores + 1)]
    df_region_base = df[df["store_id"].isin(store_ids)].copy()

    # 6-2) base_coeff 적용 → 소수점 셋째자리까지 반올림
    df_region_base["2017"] = (
        df_region_base["category"].map(base_coeff).mul(df_region_base["2017"])
    )
    df_region_base = df_region_base[["store_id", "item_id", "2017"]]

    # 6-3) 특정 행 ("{region_prefix}_1")만 골라서 specific_coeff 적용 → warehouse 행 생성
    df_src = df[df["store_id"] == f"{region_prefix}_1"].copy()
    df_region_wh = df_src.copy()
    df_region_wh["2017"] = (
        df_region_wh["category"].map(specific_coeff).mul(df_region_wh["2017"])
    )
    wh_id = f"{region_prefix}_wh"
    df_region_wh["store_id"] = wh_id
    df_region_wh = df_region_wh[["store_id", "item_id", "2017"]]

    # 6-4) base + warehouse 행 합치기
    df_out = pd.concat([df_region_base, df_region_wh], ignore_index=True)

    # 6-5) 파일명 및 저장
    output_filename = f"specific_{region_prefix}_holding.csv"
    output_path = os.path.join(output_dir, output_filename)
    df_out.to_csv(output_path, index=False)
    print(f"▶ 저장 완료: {output_path}")


# 7) CA: CA_1~CA_4, CA_wh
make_region_csv(
    region_prefix="CA", n_stores=4, base_coeff=base_coeff, specific_coeff=specific_coeff
)

# 8) TX: TX_1~TX_3, TX_wh
make_region_csv(
    region_prefix="TX", n_stores=3, base_coeff=base_coeff, specific_coeff=specific_coeff
)

# 9) WI: WI_1~WI_3, WI_wh
make_region_csv(
    region_prefix="WI", n_stores=3, base_coeff=base_coeff, specific_coeff=specific_coeff
)
