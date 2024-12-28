import os
import sys
import pandas as pd

def process_csv(model_name):
    # ディレクトリパス
    csv_dir = "./csv/"

    # ベースとなる{画像生成モデル名}.csvをロード
    base_csv_path = os.path.join(csv_dir, f"{model_name}.csv")
    if not os.path.exists(base_csv_path):
        print(f"Error: {base_csv_path} does not exist.")
        sys.exit(1)

    base_df = pd.read_csv(base_csv_path)

    # 他の{画像生成モデル名}_{評価モデル名}.csvを収集
    eval_csv_files = [
        f for f in os.listdir(csv_dir)
        if f.startswith(f"{model_name}_") and f.endswith(".csv")
    ]

    if not eval_csv_files:
        print(f"No evaluation CSV files found for model {model_name}.")
        sys.exit(1)

    # 評価モデル名ごとにデータをマージ
    merged_df = base_df.copy()

    for eval_file in sorted(eval_csv_files):  # アルファベット順にソート
        eval_csv_path = os.path.join(csv_dir, eval_file)
        eval_df = pd.read_csv(eval_csv_path)

        # {評価モデル名}列の抽出
        eval_model_name = eval_file.replace(f"{model_name}_", "").replace(".csv", "")
        eval_df = eval_df[["Filename", eval_model_name]].rename(columns={eval_model_name: eval_model_name})

        # {画像生成モデル名}.csvとマージ
        merged_df = pd.merge(merged_df, eval_df, on="Filename", how="left")

    # マージした結果を保存
    output_csv_path = os.path.join(csv_dir, f"{model_name}_merged.csv")
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged CSV saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    process_csv(model_name)
