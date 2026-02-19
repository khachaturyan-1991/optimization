import argparse
import yaml

from benchmark import Benchmark
from quantize import Quantizer
from train import Train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run training")
    parser.add_argument("--benchmark", action="store_true", help="run benchmark")
    parser.add_argument("--quantize", action="store_true", help="run quantization")
    args = parser.parse_args()

    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.train:
        trainer = Train(config)
        trainer.run()

    if args.benchmark:
        benchmarker = Benchmark(config)
        benchmarker.run()

    if args.quantize:
        quantizer = Quantizer()

if __name__ == "__main__":
    main()
