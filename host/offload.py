"""First-conv im2col GEMM tiled onto the 64x64 systolic array."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.ao.nn.intrinsic.quantized import ConvReLU2d as QConvReLU
except ImportError:
    from torch.nn.intrinsic.quantized import ConvReLU2d as QConvReLU

TILE = 64


def tiled_int32_matmul_accum(a: torch.Tensor, b: torch.Tensor, tile_fn) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dims {k} vs {k2}")
    mp = (m + TILE - 1) // TILE * TILE
    kp = (k + TILE - 1) // TILE * TILE
    np_ = (n + TILE - 1) // TILE * TILE
    ap = torch.zeros((mp, kp), dtype=torch.int32)
    bp = torch.zeros((kp, np_), dtype=torch.int32)
    ap[:m, :k] = a
    bp[:k, :n] = b
    c = torch.zeros((mp, np_), dtype=torch.int32)
    for i0 in range(0, mp, TILE):
        for j0 in range(0, np_, TILE):
            acc = torch.zeros((TILE, TILE), dtype=torch.int32)
            for k0 in range(0, kp, TILE):
                acc = acc + tile_fn(
                    ap[i0 : i0 + TILE, k0 : k0 + TILE],
                    bp[k0 : k0 + TILE, j0 : j0 + TILE],
                )
            c[i0 : i0 + TILE, j0 : j0 + TILE] = acc
    return c[:m, :n]


def cpu_tile_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.to(torch.int32), b.to(torch.int32))


def make_fpga_tile_fn(target, cs2, timeout_s=120.0, timeout_read_s=60.0):
    def tile_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != (TILE, TILE) or b.shape != (TILE, TILE):
            raise ValueError(f"expected {TILE}x{TILE}, got {tuple(a.shape)} {tuple(b.shape)}")
        a_np = a.detach().cpu().numpy().astype(np.int8)
        b_np = b.detach().cpu().numpy().astype(np.int8)
        cs2.load_matrices_A_B(target, a_np, b_np, timeout_s=timeout_s)
        cs2.run_systolic_ip(target, timeout_s=timeout_s)
        c_np = cs2.read_matrix_c(
            target,
            rows=TILE,
            cols=TILE,
            signed=True,
            timeout_s=timeout_read_s,
            fix_layout=True,
        )
        return torch.from_numpy(np.asarray(c_np, dtype=np.int32))

    return tile_fn


def _im2col_int_unfold(qx, kernel_size, padding, stride):
    x_zp = int(qx.q_zero_point())
    x_ctr = qx.int_repr().to(torch.int32) - x_zp
    cols = F.unfold(x_ctr.float(), kernel_size=kernel_size, padding=padding, stride=stride)
    return cols.transpose(1, 2).to(torch.int64)


def first_conv_via_matmul_offload(qx, layer, tile_fn):
    is_fused = isinstance(layer, QConvReLU)
    x_scale = float(qx.q_scale())
    w_q = layer.weight()
    out_scale = float(layer.scale)
    out_zp = int(layer.zero_point)
    bias = layer.bias()
    kh, kw = layer.kernel_size
    p_tuple = layer.padding
    s_tuple = layer.stride
    bsz, _, h, w = qx.shape
    cols = _im2col_int_unfold(qx, kernel_size=layer.kernel_size, padding=p_tuple, stride=s_tuple)
    n_patches = cols.shape[1]
    k_elems = cols.shape[2]
    w_int = w_q.int_repr().to(torch.int8).to(torch.int64)
    w_flat = w_int.reshape(w_int.shape[0], -1)
    a_c = cols.reshape(bsz * n_patches, k_elems).to(torch.int32)
    if w_q.qscheme() == torch.per_channel_affine:
        w_zp = w_q.q_per_channel_zero_points().to(torch.int64).view(-1, 1)
        w_sc_vec = w_q.q_per_channel_scales().to(dtype=torch.float32).view(1, -1)
        w_c = (w_flat - w_zp).to(torch.int32)
        comp_scale = float(x_scale) * w_sc_vec
    elif w_q.qscheme() == torch.per_tensor_affine:
        w_zp = int(w_q.q_zero_point())
        w_c = (w_flat - w_zp).to(torch.int32)
        comp_scale = float(x_scale * w_q.q_scale())
    else:
        raise NotImplementedError(f"Unsupported weight qscheme: {w_q.qscheme()}")
    raw_acc = tiled_int32_matmul_accum(a_c, w_c.t(), tile_fn).to(torch.int64)
    if bias is not None and bias.dtype not in (torch.float32, torch.float64):
        raw_acc = raw_acc + bias.to(torch.int64).view(1, -1)
        y_real = raw_acc.to(torch.float32) * comp_scale
    elif bias is not None:
        y_real = raw_acc.to(torch.float32) * comp_scale + bias.to(torch.float32).view(1, -1)
    else:
        y_real = raw_acc.to(torch.float32) * comp_scale
    hq = (h + 2 * p_tuple[0] - kh) // s_tuple[0] + 1
    wq = (w + 2 * p_tuple[1] - kw) // s_tuple[1] + 1
    y_real = (
        y_real.view(bsz, n_patches, layer.out_channels)
        .view(bsz, hq, wq, layer.out_channels)
        .permute(0, 3, 1, 2)
    )
    if is_fused:
        y_real = torch.relu(y_real)
    return torch.quantize_per_tensor(y_real, scale=out_scale, zero_point=out_zp, dtype=torch.quint8)
