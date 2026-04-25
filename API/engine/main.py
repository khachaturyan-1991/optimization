"""CLI entry point."""

import argparse
import yaml

try:
    from API.engine.structured_logging import configure_json_logging, log_event
except ModuleNotFoundError:
    from structured_logging import configure_json_logging, log_event


def log(msg: str):
    log_event("message", message=msg)


def _workflow_name(args: argparse.Namespace) -> str:
    """Return a stable workflow name for the run directory."""
    selected = [
        name
        for name in ("train", "benchmark", "quantize", "prune")
        if getattr(args, name)
    ]
    return selected[0] if len(selected) == 1 else "run"


def main():
    """
    Parse CLI args and run requested workflows.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="path to config file")
    parser.add_argument("--train", action="store_true", help="start training")
    parser.add_argument("--benchmark", action="store_true", help="start benchmarking")
    parser.add_argument("--quantize", action="store_true", help="start quantization")
    parser.add_argument("--prune", action="store_true", help="start pruning")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    configure_json_logging(config, workflow=_workflow_name(args))

    if args.train:
        from train import Train

        trainer = Train(config)
        trainer.run()

    if args.benchmark:
        from benchmark import Benchmark

        benchmarker = Benchmark(config)
        benchmarker.run()

    if args.quantize:
        from quantize import Quantizer

        quantizer = Quantizer(config)
        quantizer.run()

    if args.prune:
        from prune import prune_with_config

        prune_with_config(config)


if __name__ == "__main__":
    main()
