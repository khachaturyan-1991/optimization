"""CLI entry point."""

import argparse
import yaml

from benchmark import Benchmark
from quantize import Quantizer
from train import Train


def main():
    """
    Parse CLI args and run requested workflows.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="path to config file")
    parser.add_argument("--train", action="store_true", help="start training")
    parser.add_argument("--benchmark", action="store_true", help="start benchmarking")
    parser.add_argument("--quantize", action="store_true", help="start quantization")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.train:
        trainer = Train(config)
        trainer.run()

    if args.benchmark:
        benchmarker = Benchmark(config)
        benchmarker.run()

    if args.quantize:
        quantizer = Quantizer(config)
        quantizer.run()


if __name__ == "__main__":
    main()
