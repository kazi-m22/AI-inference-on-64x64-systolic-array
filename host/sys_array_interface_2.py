"""Host interface for CW305: BRAM A/B load, systolic IP, BRAM C readback."""

import time

import numpy as np

REG_MYREG1 = 0x0C
REG_MYREG2 = 0x0D
REG_ADDR_REG = 0x0F
REG_STAT_REG = 0x10
REG_CRYPT_GO = 0x05
REG_CRYPT_CIPHEROUT = 0x09

STATUS_LOAD = 0
STATUS_RDA = 1
STATUS_RDB = 2
STATUS_START_IP = 3
STATUS_RDC = 4

BYTECNT_BITS = 7
CHUNK_FWSAFE = 47
XFER_BYTES = 128
XFER_WORDS = 32
BRAM_AB_BYTES = 4096
BRAM_C_WORDS = 4096
ROWS = 64
COLS = 64


def write_bytes(dev, reg, data, chunk=CHUNK_FWSAFE):
    old = dev.bytecount_size
    dev.bytecount_size = 0
    try:
        off = 0
        while off < len(data):
            part = data[off : off + chunk]
            addr = (reg << BYTECNT_BITS) | off
            dev.fpga_write(addr, list(part))
            off += len(part)
    finally:
        dev.bytecount_size = old


def write_reg16(dev, reg, value16):
    data = int(value16 & 0xFFFF).to_bytes(2, "little")
    write_bytes(dev, reg, data, chunk=2)


def wait_not_busy(target, timeout_s=60.0, poll_s=0.001):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if target.is_done():
            return
        time.sleep(poll_s)
    raise TimeoutError("FPGA still busy after {:.1f}s".format(timeout_s))


def trigger_go(target, timeout_s=60.0):
    target.fpga_write(REG_CRYPT_GO, [1])
    wait_not_busy(target, timeout_s=timeout_s)


def read_cipherout_bytes(target, length=XFER_BYTES):
    raw = list(target.fpga_read(REG_CRYPT_CIPHEROUT, length))
    return bytes(raw[::-1])


def int8_matrix_to_bytes(mat):
    mat = np.asarray(mat, dtype=np.int8)
    if mat.size != ROWS * COLS:
        raise ValueError("Matrix must have {} elements (e.g. {}x{})".format(ROWS * COLS, ROWS, COLS))
    return mat.reshape(-1).tobytes(order="C")


def pad128(data):
    if len(data) > XFER_BYTES:
        raise ValueError("Chunk larger than 128 bytes")
    return data + bytes(XFER_BYTES - len(data))


def payload_for_rtl_msb_first(chunk_128):
    if len(chunk_128) != XFER_BYTES:
        raise ValueError("Chunk must be 128 bytes")
    return chunk_128[::-1]


def load_chunk(target, base_byte, a_chunk_128, b_chunk_128, timeout_s=60.0):
    write_reg16(target, REG_ADDR_REG, base_byte)
    write_reg16(target, REG_STAT_REG, STATUS_LOAD)
    write_bytes(target, REG_MYREG1, a_chunk_128)
    write_bytes(target, REG_MYREG2, b_chunk_128)
    trigger_go(target, timeout_s=timeout_s)


def load_full_A_B(target, A_bytes, B_bytes, timeout_s=60.0):
    if len(A_bytes) != BRAM_AB_BYTES or len(B_bytes) != BRAM_AB_BYTES:
        raise ValueError("A and B must be {} bytes each".format(BRAM_AB_BYTES))
    for base in range(0, BRAM_AB_BYTES, XFER_BYTES):
        a_chunk = payload_for_rtl_msb_first(pad128(A_bytes[base : base + XFER_BYTES]))
        b_chunk = payload_for_rtl_msb_first(pad128(B_bytes[base : base + XFER_BYTES]))
        load_chunk(target, base, a_chunk, b_chunk, timeout_s=timeout_s)


def load_matrices_A_B(target, A, B, timeout_s=60.0):
    load_full_A_B(target, int8_matrix_to_bytes(A), int8_matrix_to_bytes(B), timeout_s=timeout_s)


def run_systolic_ip(target, timeout_s=60.0):
    write_reg16(target, REG_STAT_REG, STATUS_START_IP)
    trigger_go(target, timeout_s=timeout_s)


def read_bram_a_128(target, base_byte, timeout_s=60.0):
    write_reg16(target, REG_ADDR_REG, base_byte)
    write_reg16(target, REG_STAT_REG, STATUS_RDA)
    trigger_go(target, timeout_s=timeout_s)
    return read_cipherout_bytes(target, XFER_BYTES)


def read_bram_b_128(target, base_byte, timeout_s=60.0):
    write_reg16(target, REG_ADDR_REG, base_byte)
    write_reg16(target, REG_STAT_REG, STATUS_RDB)
    trigger_go(target, timeout_s=timeout_s)
    return read_cipherout_bytes(target, XFER_BYTES)


def words_from_cipherout_chunk(raw_128, signed=True):
    """Decode 32 x 32-bit words from a 128 B cipherout block (status=4 read C)."""
    if len(raw_128) != XFER_BYTES:
        raise ValueError("Expected 128 bytes")
    words = []
    for i in range(0, XFER_BYTES, 4):
        chunk = raw_128[i : i + 4]
        words.append(int.from_bytes(chunk, byteorder="big", signed=signed))
    words.reverse()
    return words


def read_bram_c_128(target, base_word, signed=True, timeout_s=60.0):
    write_reg16(target, REG_ADDR_REG, base_word)
    write_reg16(target, REG_STAT_REG, STATUS_RDC)
    trigger_go(target, timeout_s=timeout_s)
    raw = read_cipherout_bytes(target, XFER_BYTES)
    return words_from_cipherout_chunk(raw, signed=signed)


def read_full_bram_c(target, signed=True, timeout_s=60.0):
    words = []
    for base in range(0, BRAM_C_WORDS, XFER_WORDS):
        words.extend(read_bram_c_128(target, base, signed=signed, timeout_s=timeout_s))
    return words[:BRAM_C_WORDS]


def fix_c_matrix_columns(matrix, block_cols=XFER_WORDS):
    """Undo BRAM C column reversal within each block_cols-wide slice (e.g. 32 for 64-wide).

    Systolic / pack order stores columns as [31..0] and [63..32] per row half; this maps
  them back to [0..31] and [32..63] for NumPy row-major C[i,j].
    """
    mat = np.asarray(matrix)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-D")
    rows, cols = mat.shape
    if cols % block_cols != 0:
        return mat
    parts = [np.fliplr(mat[:, c : c + block_cols]) for c in range(0, cols, block_cols)]
    return np.hstack(parts)


def read_matrix_c(target, rows=ROWS, cols=COLS, signed=True, timeout_s=60.0, fix_layout=True):
    n = rows * cols
    if n > BRAM_C_WORDS:
        raise ValueError("Matrix larger than BRAM C ({} words)".format(BRAM_C_WORDS))
    dtype = np.int32 if signed else np.uint32
    vals = np.array(read_full_bram_c(target, signed=signed, timeout_s=timeout_s)[:n], dtype=dtype)
    mat = vals.reshape(rows, cols)
    if fix_layout:
        mat = fix_c_matrix_columns(mat, block_cols=min(XFER_WORDS, cols))
    return mat


def read_full_bram_a(target, timeout_s=60.0):
    data = bytearray()
    for base in range(0, BRAM_AB_BYTES, XFER_BYTES):
        data.extend(read_bram_a_128(target, base, timeout_s=timeout_s))
    return bytes(data[:BRAM_AB_BYTES])


def read_full_bram_b(target, timeout_s=60.0):
    data = bytearray()
    for base in range(0, BRAM_AB_BYTES, XFER_BYTES):
        data.extend(read_bram_b_128(target, base, timeout_s=timeout_s))
    return bytes(data[:BRAM_AB_BYTES])


def bytes_to_int8_matrix(data_bytes):
    if len(data_bytes) != BRAM_AB_BYTES:
        raise ValueError("Expected {} bytes".format(BRAM_AB_BYTES))
    return np.frombuffer(data_bytes, dtype=np.int8).reshape(ROWS, COLS)


def read_matrix_a(target, timeout_s=60.0):
    return bytes_to_int8_matrix(read_full_bram_a(target, timeout_s=timeout_s))


def read_matrix_b(target, timeout_s=60.0):
    return bytes_to_int8_matrix(read_full_bram_b(target, timeout_s=timeout_s))


def verify_bram_ab(target, A_expected, B_expected, timeout_s=60.0):
    ok_a, ok_b, a_read, b_read = verify_bram_ab_bytes(
        target,
        int8_matrix_to_bytes(A_expected),
        int8_matrix_to_bytes(B_expected),
        timeout_s=timeout_s,
    )
    return ok_a, ok_b, bytes_to_int8_matrix(a_read), bytes_to_int8_matrix(b_read)


def make_address_ramp_bytes(length=BRAM_AB_BYTES):
    return bytes(i & 0xFF for i in range(length))


def make_address_ramp_A_B_bytes():
    ramp = make_address_ramp_bytes(BRAM_AB_BYTES)
    return ramp, ramp


def make_address_ramp_matrices():
    """A[addr]=addr, B[addr]=2*addr (mod 256), as 64x64 int8 matrices."""
    a = (np.arange(ROWS * COLS, dtype=np.uint32) & 0xFF).astype(np.int8).reshape(ROWS, COLS)
    b = ((np.arange(ROWS * COLS, dtype=np.uint32) * 2) & 0xFF).astype(np.int8).reshape(ROWS, COLS)
    return a, b


def matrices_address_ramp():
    """A and B both use address=i mod 256 (same ramp in each BRAM)."""
    a, _ = make_address_ramp_matrices()
    return a, a.copy()


def verify_bram_ab_bytes(target, A_bytes, B_bytes, timeout_s=60.0):
    load_full_A_B(target, A_bytes, B_bytes, timeout_s=timeout_s)
    a_read = read_full_bram_a(target, timeout_s=timeout_s)
    b_read = read_full_bram_b(target, timeout_s=timeout_s)
    return a_read == A_bytes, b_read == B_bytes, a_read, b_read


def first_mismatch_index(expected, actual):
    n = min(len(expected), len(actual))
    for i in range(n):
        if expected[i] != actual[i]:
            return i
    if len(expected) != len(actual):
        return n
    return None
