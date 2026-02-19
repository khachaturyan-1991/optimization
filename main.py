import yaml
from train import Train


def main():

    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trainer = Train(config)
    trainer.run()


if __name__ == "__main__":
    main()
