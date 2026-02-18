import yaml
from data_loader import DataLoder


def main():
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DataLoder(config)
    _, test_loader = loader.get_dataloaders()

    for images, labels in test_loader:
        print("images shape:", tuple(images.shape))
        print("labels shape:", tuple(labels.shape))
        break


if __name__ == "__main__":
    main()
