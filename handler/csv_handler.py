import pandas as pd


class CSVHandler:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def read(self):
        """CSV 파일을 읽어서 DataFrame으로 저장"""
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"[INFO] Loaded CSV file: {self.filepath}")
            return self.data
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.filepath}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {e}")
            return None

    def write(self, output_path: str = None):
        """현재 데이터를 CSV 파일로 저장"""
        if self.data is None:
            print("[WARNING] No data to write.")
            return

        path = output_path if output_path else self.filepath
        try:
            self.data.to_csv(path, index=False)
            print(f"[INFO] CSV written to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV: {e}")
