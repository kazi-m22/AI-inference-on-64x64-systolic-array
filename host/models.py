"""Quantized AlexNet, ResNet-18, and VGG-16 skeletons used with the INT8 checkpoints."""
from __future__ import annotations

import pickle

import torch
import torch.nn as nn
import torch.quantization as tq
from torch.ao.nn.quantized import FloatFunctional


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def fuse_alexnet(m):
    tq.fuse_modules(m.features, ["0", "1"], inplace=True)
    tq.fuse_modules(m.features, ["3", "4"], inplace=True)
    tq.fuse_modules(m.features, ["6", "7"], inplace=True)
    tq.fuse_modules(m.features, ["8", "9"], inplace=True)
    tq.fuse_modules(m.features, ["10", "11"], inplace=True)
    tq.fuse_modules(m.classifier, ["1", "2"], inplace=True)
    tq.fuse_modules(m.classifier, ["4", "5"], inplace=True)
    return m


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.skip_add = FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        return self.relu2(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def fuse_resnet18(m):
    tq.fuse_modules(m, ["conv1", "bn1", "relu"], inplace=True)
    for layer in [m.layer1, m.layer2, m.layer3, m.layer4]:
        for block in layer:
            tq.fuse_modules(block, ["conv1", "bn1", "relu1"], inplace=True)
            tq.fuse_modules(block, ["conv2", "bn2"], inplace=True)
            if block.downsample is not None:
                tq.fuse_modules(block.downsample, ["0", "1"], inplace=True)
    return m


VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        in_ch = 3
        for x in VGG16_CFG:
            if x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_ch, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_ch = x
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def fuse_vgg16(m):
    feat_fuses = []
    i = 0
    while i < len(m.features):
        if (
            i + 2 < len(m.features)
            and isinstance(m.features[i], nn.Conv2d)
            and isinstance(m.features[i + 1], nn.BatchNorm2d)
            and isinstance(m.features[i + 2], nn.ReLU)
        ):
            feat_fuses.append([str(i), str(i + 1), str(i + 2)])
            i += 3
        else:
            i += 1
    if feat_fuses:
        tq.fuse_modules(m.features, feat_fuses, inplace=True)
    tq.fuse_modules(m.classifier, [["0", "1"], ["3", "4"]], inplace=True)
    return m


class QuantWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


def pick_quantized_engine(requested="auto"):
    supported = list(torch.backends.quantized.supported_engines)
    preference = ("fbgemm", "onednn", "qnnpack")
    if requested == "auto":
        for c in preference:
            if c in supported:
                return c
        return supported[0]
    if requested in supported:
        return requested
    for c in preference:
        if c in supported:
            return c
    raise RuntimeError(f"No usable quantized backend. Supported: {supported}")


def _float_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "alexnet":
        return fuse_alexnet(AlexNet(num_classes=num_classes).eval())
    if arch == "resnet18":
        return fuse_resnet18(ResNet18(num_classes=num_classes).eval())
    if arch == "vgg16":
        return fuse_vgg16(VGG16(num_classes=num_classes).eval())
    raise ValueError(f"unknown arch {arch}")


def build_skeleton_int8(arch: str, num_classes: int, backend: str) -> nn.Module:
    torch.backends.quantized.engine = backend
    eager = QuantWrap(_float_model(arch, num_classes).eval()).eval()
    eager.qconfig = tq.get_default_qconfig(backend)
    prep = tq.prepare(eager, inplace=False).eval()
    prep(torch.randn(1, 3, 32, 32))
    return tq.convert(prep, inplace=False).eval()


def load_int8_model(checkpoint: str, arch: str, num_classes: int, backend: str, device="cpu"):
    model = build_skeleton_int8(arch, num_classes, backend)
    try:
        sd = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(checkpoint, map_location=device)
    except pickle.UnpicklingError:
        sd = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def first_conv_layer(model, arch: str):
    if arch in ("alexnet", "vgg16"):
        return model.model.features[0]
    if arch == "resnet18":
        return model.model.conv1
    raise ValueError(arch)


def forward_after_first_conv(model, arch: str, x):
    if arch in ("alexnet", "vgg16"):
        for m in model.model.features[1:]:
            x = m(x)
        x = torch.flatten(x, 1)
        x = model.model.classifier(x)
        return model.dequant(x)
    if arch == "resnet18":
        x = model.model.layer1(x)
        x = model.model.layer2(x)
        x = model.model.layer3(x)
        x = model.model.layer4(x)
        x = model.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.model.fc(x)
        return model.dequant(x)
    raise ValueError(arch)
