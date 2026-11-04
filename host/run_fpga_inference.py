"""INT8 inference with first conv offloaded to the CW305 64x64 systolic array.

Examples (from the repository root):

  python host/run_fpga_inference.py --list
  python host/run_fpga_inference.py --config cifar10-alexnet --dataflow os --n 10
  python host/run_fpga_inference.py --config cifar10-alexnet --cpu --n 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from models import (  # noqa: E402
    first_conv_layer,
    forward_after_first_conv,
    load_int8_model,
    pick_quantized_engine,
)
from offload import cpu_tile_fn, first_conv_via_matmul_offload, make_fpga_tile_fn  # noqa: E402

FPGA_ID = "100t"
CLKGEN_HZ = 100e6
FPGA_TIMEOUT_S = 120.0
FPGA_TIMEOUT_READ_S = 60.0

CIFAR10_NAMES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}
GTSRB_NAMES = {
    0: "limit_20", 1: "limit_30", 2: "limit_50", 3: "limit_60", 4: "limit_70",
    5: "limit_80", 6: "end_limit_80", 7: "limit_100", 8: "limit_120",
    9: "no_passing", 10: "no_pass_truck", 11: "right_of_way", 12: "priority",
    13: "yield", 14: "stop", 15: "no_vehicles", 16: "no_trucks", 17: "no_entry",
    18: "caution", 19: "curve_left", 20: "curve_right", 21: "double_curve",
    22: "bumpy", 23: "slippery", 24: "narrows", 25: "road_work",
    26: "signals", 27: "pedestrian", 28: "children", 29: "bicycles",
    30: "ice_snow", 31: "animals", 32: "end_limits", 33: "turn_right",
    34: "turn_left", 35: "ahead", 36: "ahead_right", 37: "ahead_left",
    38: "keep_right", 39: "keep_left", 40: "roundabout", 41: "end_no_pass",
    42: "end_no_pass_trk",
}
MNIST_NAMES = {i: str(i) for i in range(10)}

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
HALF_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
HALF_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

DATAFLOWS = {
    "os": ROOT / "bitfiles" / "100mhz_os.bit",
    "is": ROOT / "bitfiles" / "100mhz_is.bit",
    "ws": ROOT / "bitfiles" / "100mhz_ws.bit",
}

CONFIGS = {
    "cifar10-alexnet": {
        "arch": "alexnet", "num_classes": 10, "norm": "cifar", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "cifar10" / "quantized_alexnet_cifar10.pth",
        "samples": ROOT / "samples" / "cifar10" / "cifar10_quantized_100_per_class.npz",
    },
    "cifar10-resnet18": {
        "arch": "resnet18", "num_classes": 10, "norm": "half", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "cifar10" / "cifar10_quantized_resnet18.pth",
        "samples": ROOT / "samples" / "cifar10" / "cifar10_quantized_100_per_class.npz",
    },
    "cifar10-vgg16": {
        "arch": "vgg16", "num_classes": 10, "norm": "half", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "cifar10" / "cifar10_quantized_vgg16.pth",
        "samples": ROOT / "samples" / "cifar10" / "cifar10_quantized_100_per_class.npz",
    },
    "custom-cifar10-alexnet": {
        "arch": "alexnet", "num_classes": 10, "norm": "half", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "cifar10" / "quantized_alexnet_custom_cifar10_weights.pth",
        "samples": ROOT / "samples" / "custom_cifar10" / "quantized_correct_100_per_class.npz",
    },
    "custom-cifar10-resnet18": {
        "arch": "resnet18", "num_classes": 10, "norm": "half", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "cifar10" / "custom_cifar10_quantized_resnet18.pth",
        "samples": ROOT / "samples" / "custom_cifar10" / "quantized_correct_100_per_class.npz",
    },
    "shuffled-cifar10-alexnet": {
        "arch": "alexnet", "num_classes": 10, "norm": "half", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "shuffled_cifar10" / "shuffeled_cifar10_quantized_alexnet_cifar10_2.pth",
        "samples": ROOT / "samples" / "shuffled_cifar10" / "shuffled_cifar10_correct_100.npz",
        "note": "NPZ labels are shuffled model IDs; see samples/shuffled_cifar10/label_map.txt",
    },
    "gtsrb-alexnet": {
        "arch": "alexnet", "num_classes": 43, "norm": "half", "names": GTSRB_NAMES,
        "weights": ROOT / "weights" / "gtsrb" / "gtsrb_quantized_alexnet_orignal_label_no_shuffle.pth",
        "samples": ROOT / "samples" / "gtsrb" / "gtsrb_quantized_10_per_class.npz",
    },
    "gtsrb-resnet18": {
        "arch": "resnet18", "num_classes": 43, "norm": "half", "names": GTSRB_NAMES,
        "weights": ROOT / "weights" / "gtsrb" / "gtsrb_quantized_resnet18_orignal_label_no_shuffle.pth",
        "samples": ROOT / "samples" / "gtsrb" / "gtsrb_quantized_10_per_class.npz",
    },
    "gtsrb-vgg16": {
        "arch": "vgg16", "num_classes": 43, "norm": "half", "names": GTSRB_NAMES,
        "weights": ROOT / "weights" / "gtsrb" / "gtsrb_quantized_vgg16_orignal_label_no_shuffle.pth",
        "samples": ROOT / "samples" / "gtsrb" / "gtsrb_quantized_10_per_class.npz",
    },
    "shuffled-gtsrb-alexnet": {
        "arch": "alexnet", "num_classes": 43, "norm": "half", "names": GTSRB_NAMES,
        "weights": ROOT / "weights" / "gtsrb" / "shuffled_gtsrb_quantized_alexnet.pth",
        "samples": ROOT / "samples" / "gtsrb" / "gtsrb_quantized_10_per_class.npz",
        "note": "Weights use shuffled IDs (samples/gtsrb/gtsrb_shuffled_label_map.txt). Default NPZ is original-label.",
    },
    "mnist-alexnet": {
        "arch": "alexnet", "num_classes": 10, "norm": "half", "names": MNIST_NAMES,
        "weights": ROOT / "weights" / "mnist" / "mnist_quantized_alexnet.pth",
        "samples": ROOT / "samples" / "mnist" / "mnist_alexnet_int8_correct_100per_class.npz",
    },
    "mnist-resnet18": {
        "arch": "resnet18", "num_classes": 10, "norm": "half", "names": MNIST_NAMES,
        "weights": ROOT / "weights" / "mnist" / "mnist_quantized_resnet18.pth",
        "samples": ROOT / "samples" / "mnist" / "mnist_alexnet_int8_correct_100per_class.npz",
    },
    "mnist-vgg16": {
        "arch": "vgg16", "num_classes": 10, "norm": "half", "names": MNIST_NAMES,
        "weights": ROOT / "weights" / "mnist" / "mnist_quantized_vgg16.pth",
        "samples": ROOT / "samples" / "mnist" / "mnist_alexnet_int8_correct_100per_class.npz",
    },
    "cifar10-75-25-alexnet": {
        "arch": "alexnet", "num_classes": 10, "norm": "half", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "cifar10" / "cifar10_75_25_quantized_alexnet_cifar10.pth",
        "samples": ROOT / "samples" / "cifar10_75_25" / "cifar10_25extract_quantized_100_per_class.npz",
    },
    "frog-ship-alexnet": {
        "arch": "alexnet", "num_classes": 10, "norm": "half", "names": CIFAR10_NAMES,
        "weights": ROOT / "weights" / "cifar10" / "quantized_alexnet_cifar10_imagenet_frog_ship.pth",
        "samples": ROOT / "samples" / "mixed_frog_ship" / "mixed_val_set.npz",
    },
}


def dataset_row_to_model_input(img: np.ndarray, norm: str) -> torch.Tensor:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        x = arr.astype(np.float32).copy()
        if x.shape[0] == 1:
            x = np.repeat(x, 3, axis=0)
        if np.issubdtype(arr.dtype, np.integer) or float(np.nanmax(x)) > 1.5:
            x = x / 255.0
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        x = arr.astype(np.float32)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        if np.issubdtype(arr.dtype, np.integer) or float(np.nanmax(x)) > 1.5:
            x = x / 255.0
        x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image shape {arr.shape}")
    if x.shape[1] != 32 or x.shape[2] != 32:
        # nearest upsample/down to 32x32 for AlexNet/ResNet/VGG stems used here
        t = torch.from_numpy(x).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
        x = t.squeeze(0).numpy()
    mean, std = (CIFAR_MEAN, CIFAR_STD) if norm == "cifar" else (HALF_MEAN, HALF_STD)
    for c in range(3):
        x[c] = (x[c] - mean[c]) / std[c]
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def load_npz_pairs(path: Path, n: int | None):
    z = np.load(path)
    images = np.asarray(z["images"])
    labels = np.asarray(z["labels"], dtype=np.int64).reshape(-1)
    if images.shape[0] != labels.shape[0]:
        raise RuntimeError(f"images {images.shape} vs labels {labels.shape} in {path}")
    if n is not None and n > 0:
        images, labels = images[:n], labels[:n]
    return images, labels


def init_hardware(bitfile: Path):
    import chipwhisperer as cw
    import sys_array_interface_2 as cs2

    if not bitfile.is_file():
        raise FileNotFoundError(bitfile)
    scope = cw.scope()
    scope.gain.db = 25
    scope.adc.samples = 129
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.clock.clkgen_freq = CLKGEN_HZ
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    scope.io.tio1 = "serial_rx"
    scope.io.tio2 = "serial_tx"
    scope.io.hs2 = "clkgen"
    target = cw.target(scope, cw.targets.CW305, fpga_id=FPGA_ID, force=True, bsfile=str(bitfile))
    target.vccint_set(1.0)
    target.pll.pll_enable_set(True)
    target.pll.pll_outenable_set(False, 0)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outenable_set(False, 2)
    target.pll.pll_outfreq_set(CLKGEN_HZ, 1)
    target.clkusbautooff = True
    target.clksleeptime = 1
    scope.clock.adc_src = "clkgen_x4"
    for _ in range(5):
        scope.clock.reset_adc()
        time.sleep(1)
        if scope.clock.adc_locked:
            break
    if not scope.clock.adc_locked:
        raise RuntimeError("ADC failed to lock")
    scope.io.hs2 = "clkgen"
    print(f"programmed {bitfile}", flush=True)
    print("ADC locked, hs2=clkgen (clean inference, no glitch clock)", flush=True)
    return scope, target, cs2


def close_hardware(scope, target) -> None:
    for obj in (target, scope):
        if obj is None:
            continue
        try:
            obj.dis()
        except Exception:
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FPGA first-conv INT8 inference on CW305 64x64")
    p.add_argument("--config", choices=sorted(CONFIGS), default="cifar10-alexnet")
    p.add_argument("--dataflow", choices=sorted(DATAFLOWS), default="os", help="OS / IS / WS bitfile")
    p.add_argument("--n", type=int, default=10, help="Number of images (0 = all in the NPZ)")
    p.add_argument("--cpu", action="store_true", help="CPU tiled int32 matmul instead of the FPGA")
    p.add_argument("--list", action="store_true", help="Print configs and exit")
    p.add_argument("--bitfile", type=str, default="", help="Override bitfile path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        print("Available --config values:")
        for name, cfg in CONFIGS.items():
            w = cfg["weights"]
            s = cfg["samples"]
            extra = f"  # {cfg['note']}" if cfg.get("note") else ""
            print(f"  {name:28s}  {cfg['arch']:9s}  C={cfg['num_classes']:<3}  weights={'yes' if w.is_file() else 'MISSING'}  samples={'yes' if s.is_file() else 'MISSING'}{extra}")
        print("\nDataflows (--dataflow): os, is, ws")
        return 0

    cfg = CONFIGS[args.config]
    if not cfg["weights"].is_file():
        print(f"missing weights: {cfg['weights']}", file=sys.stderr)
        return 2
    if not cfg["samples"].is_file():
        print(f"missing samples: {cfg['samples']}", file=sys.stderr)
        return 2

    backend = pick_quantized_engine("auto")
    print(f"config={args.config}  arch={cfg['arch']}  backend={backend}  dataflow={args.dataflow}")
    model = load_int8_model(str(cfg["weights"]), cfg["arch"], cfg["num_classes"], backend)
    images, labels = load_npz_pairs(cfg["samples"], None if args.n <= 0 else args.n)
    print(f"images {images.shape} from {cfg['samples'].name}")
    names = cfg["names"]
    layer = first_conv_layer(model, cfg["arch"])

    scope = target = None
    if args.cpu:
        tile_fn = cpu_tile_fn
        print("using CPU tile matmul")
    else:
        bitfile = Path(args.bitfile) if args.bitfile else DATAFLOWS[args.dataflow]
        scope, target, cs2 = init_hardware(bitfile)
        tile_fn = make_fpga_tile_fn(target, cs2, FPGA_TIMEOUT_S, FPGA_TIMEOUT_READ_S)

    ok = 0
    try:
        with torch.inference_mode():
            for i, (img, y) in enumerate(zip(images, labels)):
                xb = dataset_row_to_model_input(img, cfg["norm"])
                qx = model.quant(xb)
                conv1 = first_conv_via_matmul_offload(qx, layer, tile_fn)
                logits = forward_after_first_conv(model, cfg["arch"], conv1)
                pred = int(logits.argmax(dim=-1).item())
                hit = pred == int(y)
                ok += int(hit)
                yname = names.get(int(y), str(int(y)))
                pname = names.get(pred, str(pred))
                print(f"  [{i + 1:4d}/{len(labels)}] true={int(y):2d} ({yname})  pred={pred:2d} ({pname})  {'ok' if hit else 'miss'}", flush=True)
    finally:
        close_hardware(scope, target)

    acc = 100.0 * ok / max(1, len(labels))
    print(f"accuracy {ok}/{len(labels)} = {acc:.1f}%")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
