"""MobileNetV2 model definition with quantization helpers."""

import logging
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub


class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        self.skip_add = nn.quantized.FloatFunctional()

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        return self.conv(x)


def _make_divisible(v, divisor=8, min_value=None):
    """Ensure channels are divisible by a given divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):
    def __init__(self, cfg="config.yml"):
        """Build MobileNetV2 from config dict."""
        super().__init__()

        width_mult = cfg['width_multiplier']
        num_classes = cfg['num_classes']
        settings = cfg['inverted_residual_setting']

        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(cfg['last_channel'] * max(1.0, width_mult))

        self.features = [ConvBNReLU(3, input_channel, stride=1)]

        for t, c, n, s in settings:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        self.features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*self.features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        ckpt_path = cfg.get("checkpoint_path")
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_checkpoint(ckpt_path)

    def get_layer_names(self):
        """Return list of layer names for all named modules."""
        return [name for name, _ in self.named_modules() if name]

    def save_model(self, ckpt_path: str = "model.pt"):
        """Save JIT model to disk."""
        device = None
        for p in self.parameters():
            device = p.device
            break
        if device is None:
            for b in self.buffers():
                device = b.device
                break
        if device is None:
            device = torch.device("cpu")
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(self, dummy_input)
        if was_training:
            self.train()
        torch.jit.save(traced_model, ckpt_path)
        logging.info("Model saved to %s", ckpt_path)

    def _load_checkpoint(self, ckpt_path: str):
        """Load weights from a JIT checkpoint only."""
        if not os.path.exists(ckpt_path):
            logging.error("Checkpoint not found: %s", ckpt_path)
            return
        logging.info("Loading JIT model and extracting state_dict from %s", ckpt_path)
        jit_model = torch.jit.load(ckpt_path, map_location="cpu")
        state_dict = jit_model.state_dict()
        for k, v in list(state_dict.items()):
            if torch.is_tensor(v) and v.is_quantized:
                state_dict[k] = v.dequantize()
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            num_missing = len(incompatible.missing_keys)
            num_unexpected = len(incompatible.unexpected_keys)
            logging.error(
                "Checkpoint load partial: missing=%s unexpected=%s",
                num_missing,
                num_unexpected,
            )
        else:
            logging.info("Successfully loaded model weights.")

    def _fuse_model(self):
        """Fuse Conv+BN and Conv+BN+ReLU for quantization."""
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        if idx + 1 < len(m.conv) and type(m.conv[idx+1]) == nn.BatchNorm2d:
                            torch.ao.quantization.fuse_modules(m.conv, [str(idx), str(idx+1)], inplace=True)

    def _prepare_model(self):
            """Fuse layers and prepare model for static quantization."""
            engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "fbgemm"
            torch.backends.quantized.engine = engine
            self.qconfig = torch.ao.quantization.get_default_qconfig(engine)

            self._fuse_model()

            torch.ao.quantization.prepare(self, inplace=True)

    def forward(self, x):
        """Forward pass with quant/dequant stubs."""
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, cfg: Dict):
        super(SimpleCNN, self).__init__()
        input_channels = int(cfg.get("input_channels", 1))
        num_classes = int(cfg.get("num_classes", 10))
        self.input_size = int(cfg.get("input_size", 28))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        feature_dim = self._infer_feature_dim(input_channels)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        ckpt_path = cfg.get("checkpoint_path")
        if ckpt_path:
            self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            return
        logging.info("Loading JIT model and extracting state_dict from %s", ckpt_path)
        jit_model = torch.jit.load(ckpt_path, map_location="cpu")
        state_dict = jit_model.state_dict()
        self.load_state_dict(state_dict, strict=False)

    def _infer_feature_dim(self, input_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, self.input_size, self.input_size)
            out = self.features(dummy)
        return int(out.numel())

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def save_model(self, ckpt_path: str = "model.pt"):
        dummy_input = torch.randn(1, 1, self.input_size, self.input_size)
        self.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(self, dummy_input)
        torch.jit.save(traced_model, ckpt_path)

    def _fuse_model(self, is_qat=False):
        for m in self.modules():
            if type(m) == nn.Sequential:
                # SimpleCNN doesn't have BN, so we just pass
                pass


def get_model(cfg: Dict) -> nn.Module:
    model_name = cfg.get("name", "mobilenet_v2").lower()
    if model_name == "simple_cnn":
        return SimpleCNN(cfg)
    return MobileNetV2(cfg)
