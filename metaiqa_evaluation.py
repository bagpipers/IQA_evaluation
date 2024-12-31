import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class MetaIQAFeatureExtractor(nn.Module):
    def __init__(self, input_size=3*512*512, hidden_size=1024, output_size=512):
        super(MetaIQAFeatureExtractor, self).__init__()
        self.input_size = input_size

        # First layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.1)

        # Second layer with residual connection
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.residual_connection = nn.Linear(hidden_size, output_size)  # 調整用
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.1)

        # Output layer
        self.fc3 = nn.Linear(output_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input if necessary
        
        # First layer
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second layer with residual connection
        residual = self.residual_connection(out)  # サイズを一致させる
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out += residual  # Add residual connection

        # Output layer
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


# ディレクトリ内の画像ファイルを取得する関数
def get_image_files(directory):
    supported_formats = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    return [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in supported_formats]

# 推論して結果をCSVに書き込む関数
def evaluate_images_and_save_csv(model, image_files, root_dir, output_dir, device):
    output_csv = os.path.join(output_dir, f"{os.path.basename(root_dir.strip('/'))}_metaiqa.csv")

    # 既存のCSVファイルが存在する場合は削除
    if os.path.exists(output_csv):
        os.remove(output_csv)

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    results = []
    total_images = len(image_files)
    print(f"Total images to evaluate: {total_images}")

    with torch.no_grad():
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(root_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)  # バッチ次元を追加

            predicted_score = model(image).item()  # 推定スコア

            # 結果をリストに追加
            results.append({"Filename": img_name, "metaiqa": predicted_score})
            print(f"[{idx + 1}/{total_images}] Image: {img_name}, Predicted Score: {predicted_score}")

    # 結果をCSVに保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # 評価対象の画像ディレクトリ
    image_directory = "../IQA-PyTorch/datasets/koniq10k/1024x768"
    output_directory = "./csv/"  # 出力ディレクトリを指定

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 対象画像ファイルを取得
    image_files = get_image_files(image_directory)

    # モデルのパラメータ
    input_size = 3 * 512 * 512
    hidden_size = 1024
    output_size = 512

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの初期化とロード
    model = MetaIQAFeatureExtractor(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load("./meta_iqa_model.pth", map_location=device))

    # 画像を評価して結果をCSVに保存
    evaluate_images_and_save_csv(model, image_files, image_directory, output_directory, device)
