"""Simple structural pruning script using torch_pruning."""

import argparse
import glob
import os
import yaml
import torch
import torch_pruning as tp

from _model_loader import LoaderTorchJit


def _find_latest_checkpoint(ckpt_dir: str) -> str | None:
    pattern = os.path.join(ckpt_dir, "epoch_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _build_example_inputs(cfg: dict) -> torch.Tensor:
    """Construct example inputs from config or fall back to CIFAR-sized input."""
    cfg_model = cfg.get("model", {})
    model_name = cfg_model.get("name", "").lower()
    if model_name == "simple_cnn":
        input_channels = int(cfg_model.get("input_channels", 1))
        input_size = int(cfg_model.get("input_size", 28))
        return torch.randn(1, input_channels, input_size, input_size)

    data_cfg = cfg.get("data", {})
    input_channels = int(cfg_model.get("input_channels", data_cfg.get("input_channels", 3)))
    input_size = int(cfg_model.get("input_size", data_cfg.get("input_size", 32)))
    return torch.randn(1, input_channels, input_size, input_size)


def prune_with_config(cfg: dict) -> None:
    pruning_cfg = cfg.get("pruning", {})
    ckpt_path = pruning_cfg.get("checkpoint_path")
    if not ckpt_path:
        ckpt_dir = cfg.get("train", {}).get("ckpt_dir", "checkpoints")
        ckpt_path = _find_latest_checkpoint(ckpt_dir)

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No checkpoint found. Set pruning.checkpoint_path or ensure checkpoints exist."
        )

    model = LoaderTorchJit(ckpt_path).model
    model.to("cpu")
    example_inputs = _build_example_inputs(cfg)

    ch_sparsity = float(pruning_cfg.get("ch_sparsity", 0.3))
    if not (0.0 <= ch_sparsity < 1.0):
        raise ValueError("pruning.ch_sparsity must be in [0.0, 1.0).")
    print(f"Using pruning.ch_sparsity: {ch_sparsity}")

    def _count_params(m):
        return sum(p.numel() for p in m.parameters())

    before_params = _count_params(model)

    importance = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = []
    ignore_names = pruning_cfg.get("ignore_layers", [])
    if ignore_names:
        name_to_module = dict(model.named_modules())
        for name in ignore_names:
            module = name_to_module.get(name)
            if module is None:
                print(f"[prune] ignore_layers: '{name}' not found in model.")
                continue
            ignored_layers.append(module)
    else:
        if hasattr(model, "classifier") and len(model.classifier) > 0:
            ignored_layers.append(model.classifier[-1])

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        ch_sparsity=ch_sparsity,
        ignored_layers=ignored_layers,
    )
    pruner.step()

    after_params = _count_params(model)
    print(f"Params before: {before_params}")
    print(f"Params after : {after_params}")

    output_path = pruning_cfg.get("output_path", "pruned_simple_cnn.pt")
    torch.jit.save(model, output_path)
    print(f"Pruned model saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_mnist.yml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prune_with_config(cfg)


if __name__ == "__main__":
    main()
