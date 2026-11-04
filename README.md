# AI inference on a 64×64 systolic array

INT8 convolutional-network inference on a **ChipWhisperer CW305** FPGA (Artix-7 `100t`) with a **64×64** systolic matrix engine. The host runs the quantized PyTorch model; the **first convolution** is lowered to im2col GEMM and executed as 64×64 tiles on the FPGA. Remaining layers stay on the CPU.

Three bitstreams are included, one per dataflow:

| `--dataflow` | Bitfile | Stationarity |
|---|---|---|
| `os` | `bitfiles/100mhz_os.bit` | output-stationary |
| `is` | `bitfiles/100mhz_is.bit` | input-stationary |
| `ws` | `bitfiles/100mhz_ws.bit` | weight-stationary |

Clock: **100 MHz** (ChipWhisperer `clkgen` + CW305 PLL output 1). Host protocol: USB register/BRAM transfers in `host/sys_array_interface_2.py`.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── bitfiles/
│   ├── 100mhz_os.bit
│   ├── 100mhz_is.bit
│   └── 100mhz_ws.bit
├── host/
│   ├── run_fpga_inference.py      # main entry point
│   ├── models.py                  # INT8 AlexNet / ResNet-18 / VGG-16
│   ├── offload.py                 # im2col + 64×64 tiling
│   └── sys_array_interface_2.py   # CW305 load / start / read C
├── weights/                       # quantized state_dicts
│   ├── cifar10/
│   ├── gtsrb/
│   ├── mnist/
│   └── shuffled_cifar10/
└── samples/                       # compact NPZ evaluation sets
    ├── cifar10/
    ├── custom_cifar10/
    ├── shuffled_cifar10/
    ├── gtsrb/
    ├── mnist/
    ├── mixed_frog_ship/
    └── cifar10_75_25/
```

Full-size training corpora (ImageNet-32 shards, 5k-per-class dumps) are **not** in this repo. The NPZs under `samples/` are enough to run inference.

### Weights vs configs

| `--config` | Architecture | Classes | Weights | Samples |
|---|---|---|---|---|
| `cifar10-alexnet` | AlexNet | 10 | `weights/cifar10/quantized_alexnet_cifar10.pth` | `samples/cifar10/cifar10_quantized_100_per_class.npz` |
| `cifar10-resnet18` | ResNet-18 | 10 | `weights/cifar10/cifar10_quantized_resnet18.pth` | same CIFAR-10 NPZ |
| `cifar10-vgg16` | VGG-16 | 10 | `weights/cifar10/cifar10_quantized_vgg16.pth` | same CIFAR-10 NPZ |
| `custom-cifar10-alexnet` | AlexNet | 10 | `weights/cifar10/quantized_alexnet_custom_cifar10_weights.pth` | `samples/custom_cifar10/...` |
| `custom-cifar10-resnet18` | ResNet-18 | 10 | `weights/cifar10/custom_cifar10_quantized_resnet18.pth` | custom CIFAR-10 NPZ |
| `shuffled-cifar10-alexnet` | AlexNet | 10 | `weights/shuffled_cifar10/shuffeled_cifar10_quantized_alexnet_cifar10_2.pth` | shuffled NPZ + `label_map.txt` |
| `gtsrb-alexnet` | AlexNet | 43 | `weights/gtsrb/gtsrb_quantized_alexnet_orignal_label_no_shuffle.pth` | `samples/gtsrb/gtsrb_quantized_10_per_class.npz` |
| `gtsrb-resnet18` | ResNet-18 | 43 | `weights/gtsrb/gtsrb_quantized_resnet18_orignal_label_no_shuffle.pth` | GTSRB NPZ |
| `gtsrb-vgg16` | VGG-16 | 43 | `weights/gtsrb/gtsrb_quantized_vgg16_orignal_label_no_shuffle.pth` | GTSRB NPZ |
| `shuffled-gtsrb-alexnet` | AlexNet | 43 | `weights/gtsrb/shuffled_gtsrb_quantized_alexnet.pth` | see `gtsrb_shuffled_label_map.txt` |
| `mnist-alexnet` / `mnist-resnet18` / `mnist-vgg16` | as named | 10 | `weights/mnist/` | `samples/mnist/mnist_alexnet_int8_correct_100per_class.npz` |
| `cifar10-75-25-alexnet` | AlexNet | 10 | `weights/cifar10/cifar10_75_25_quantized_alexnet_cifar10.pth` | `samples/cifar10_75_25/...` |
| `frog-ship-alexnet` | AlexNet | 10 | `weights/cifar10/quantized_alexnet_cifar10_imagenet_frog_ship.pth` | `samples/mixed_frog_ship/mixed_val_set.npz` |

Also included: `weights/cifar10/alexnet_quantized_int8.pth` (alternate custom-CIFAR AlexNet checkpoint).

---

## Software setup

Python 3.10+ with a CPU PyTorch build that has a quantized engine (`fbgemm`, `onednn`, or `qnnpack`).

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On this lab machine the existing ChipWhisperer venv works:

```powershell
C:\Users\kmejbaulislam\ChipWhisperer\cwenv\Scripts\python.exe -m pip install -r requirements.txt
```

`chipwhisperer` is required only for FPGA runs (`--cpu` does not need USB).

---

## ChipWhisperer / CW305

Hardware:

- NewAE **CW305** target, FPGA id `100t`
- ChipWhisperer capture board (Lite or Husky) as USB programmer and clock source
- SMA/shunt as wired in the 100 MHz experiments (not required for functional inference)

**Only one process may own the USB device.** Close Jupyter kernels and other sweep scripts before starting.

What the host does (`init_hardware` in `run_fpga_inference.py`):

1. `cw.scope()` then `cw.target(..., cw.targets.CW305, fpga_id="100t", bsfile=<bitfile>, force=True)`
2. `target.vccint_set(1.0)` and enable PLL output **1** at 100 MHz
3. `scope.clock.clkgen_freq = 100e6`, `scope.clock.adc_src = "clkgen_x4"`
4. `scope.io.hs2 = "clkgen"` — **clean array clock, no glitch XOR**
5. Wait until `scope.clock.adc_locked`

Per 64×64 tile (`host/sys_array_interface_2.py`):

1. Pack int8 **A** and **B** into BRAM (`STATUS_LOAD` + `REG_CRYPT_GO`)
2. `STATUS_START_IP` + GO — systolic multiply–accumulate
3. Read int32 **C** (`STATUS_RDC`), undo the BRAM column reversal (`fix_layout=True`)

The first conv of a 3×3 / pad-1 / 64-channel stem on 32×32 is im2col **1024×27** times **27×64**. K is padded to 64, so the GEMM is **16 tiles** of 64×64. Each image therefore performs 16 FPGA matmuls.

TIO4 is the FPGA GO strobe (used if you capture power). It is not required for classification.

---

## Running inference

List every config (and whether weights/samples are present):

```powershell
cd C:\Users\kmejbaulislam\Documents\AI-inference-on-64x64-systolic-array
python host/run_fpga_inference.py --list
```

FPGA, output-stationary bitstream, 10 CIFAR-10 images:

```powershell
python host/run_fpga_inference.py --config cifar10-alexnet --dataflow os --n 10
```

Input-stationary or weight-stationary:

```powershell
python host/run_fpga_inference.py --config cifar10-resnet18 --dataflow is --n 10
python host/run_fpga_inference.py --config gtsrb-alexnet --dataflow ws --n 10
```

CPU-only check of the same tiling (no ChipWhisperer):

```powershell
python host/run_fpga_inference.py --config cifar10-alexnet --cpu --n 20
```

All images in the NPZ:

```powershell
python host/run_fpga_inference.py --config cifar10-alexnet --dataflow os --n 0
```

Override the bitfile path:

```powershell
python host/run_fpga_inference.py --config cifar10-alexnet --bitfile path\to\custom.bit
```

`--n` takes the first N rows of the NPZ. Accuracy is printed as `ok/N`.

---

## How the offload works

1. `QuantStub` produces a `quint8` tensor.
2. The fused first conv (ConvReLU or Conv-BN-ReLU) is expanded with `F.unfold` (im2col).
3. Zero-points are removed; the GEMM is **int32** accumulate of **int8** tiles.
4. Each 64×64 tile is sent to the FPGA (or `torch.matmul` with `--cpu`).
5. Scale, bias, ReLU, and requantization run on the host.
6. The rest of the INT8 network (`features[1:]` or ResNet layers) runs in PyTorch.

Shuffled-label models: NPZ labels are the **training** class IDs. See `samples/shuffled_cifar10/label_map.txt` and `samples/gtsrb/gtsrb_shuffled_label_map.txt`.
