import torch
from torch import nn
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
import sys

class MetaIQAPredictor(nn.Module):
    def __init__(self, input_size=3*512*512, hidden_size=1024, output_size=512):
        super(MetaIQAPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(),
            nn.Linear(output_size, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.model(x)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)  # Read the CSV file
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Filename from the first column
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")  # Load the image as RGB

        if self.transform:
            img = self.transform(img)  # Apply transformations

        return img, img_name

def evaluate_images_and_update_csv(model, dataset, device, input_csv, output_csv):
    model.eval()
    results = []

    with torch.no_grad():
        for img, img_name in dataset:
            img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
            predicted_score = model(img).item()  # Get the predicted score

            # Add the result to the list
            results.append((img_name, predicted_score))

    # Load the input CSV
    df = pd.read_csv(input_csv)

    # Add a new column with the predicted scores
    predicted_scores = {img_name: score for img_name, score in results}
    df["Predicted_Score"] = df["Filename"].map(predicted_scores)

    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Parameters
if len(sys.argv) < 2:
    print("Usage: python script.py <model_directory_path>")
    sys.exit(1)

model_path = sys.argv[1]  # Accept model directory path from command line
model_name = os.path.basename(model_path.rstrip('/'))  # Extract model name from path
image_dir = model_path  # Images are located in the provided model directory
input_csv = os.path.join(model_path, "input.csv")  # CSV file is in the model directory
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv", f"output_{model_name}.csv")  # Output CSV will be saved in ./csv directory

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image
    transforms.ToTensor(),  # Convert to tensor
])

# Initialize the model
input_size = 3 * 512 * 512
hidden_size = 1024
output_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetaIQAPredictor(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

# Load the model
model_file_path = os.path.join(model_path, "model.pth")  # Model file is in the model directory
model.load_state_dict(torch.load(model_file_path, map_location=device))

# Create the dataset and evaluate
dataset = ImageDataset(csv_file=input_csv, image_dir=image_dir, transform=transform)
evaluate_images_and_update_csv(model, dataset, device, input_csv, output_csv)
