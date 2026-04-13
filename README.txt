Quantization and Pruning Experiments

Short description
This repo illustrates how to train, benchmark, and prune image classification models (MobileNetV2 and SimpleCNN) on CIFAR-10 and MNIST.

Setup
- Python 3.11+ recommended
- Install dependencies (example):

  pip install torch torchvision tqdm pyyaml matplotlib pillow tensorboard mlflow torch-pruning

Usage
1. Train
- CIFAR-10 (MobileNetV2)

  python main.py --train --config configs/config.yml

- MNIST (SimpleCNN)

  python main.py --train --config configs/config_mnist.yml

2. Benchmark
- CIFAR-10

  python main.py --benchmark --config configs/config.yml

- MNIST

  python main.py --benchmark --config configs/config_mnist.yml

3. Prune
- CIFAR-10

  python main.py --prune --config configs/config.yml

- MNIST

  python main.py --prune --config configs/config_mnist.yml

Datasets
This repo uses torchvision datasets and downloads them automatically on first run.

CIFAR-10
- Set in config: data.dataset: "cifar10"
- Files are downloaded to the path in data.data_dir (default: ./DATA)
- Run training or benchmarking once to trigger the download

MNIST
- Set in config: data.dataset: "mnist"
- Files are downloaded to the path in data.data_dir (default: ./DATA)
- Run training or benchmarking once to trigger the download

Notes
- Checkpoints are saved to train.ckpt_dir (default: ./checkpoints)
- TensorBoard logs go to logs.log_dir (default: ./runs)
- MLflow logging is optional and controlled via the mlflow section in config files

Publishing on GitHub
- Commit README.txt to the repo root
- Ensure configs in ./configs are up to date with your experiment settings
