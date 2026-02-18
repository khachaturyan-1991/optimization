import torch
import yaml
from data_loader import DataLoder
from model import MobileNetV2 # Assuming your class is in model_file.py


def main():

    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MobileNetV2(cfg=config["model"]).to(device)
    model.eval()

    loader = DataLoder(config)
    _, test_loader = loader.get_dataloaders()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        print("Images shape:", tuple(images.shape))

        with torch.no_grad():
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

        print("Predicted indices:", predicted.tolist())
        print("Actual labels:   ", labels.tolist())

        break


if __name__ == "__main__":
    main()
