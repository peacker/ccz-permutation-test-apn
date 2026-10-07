"""ctypes bindings for _fast.c (compiled on first import with the system C compiler)."""

import ctypes
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_fast.c")
_BUILD = os.environ.get("APNSEARCH_BUILD", os.path.join(os.path.expanduser("~"), ".cache", "apnsearch"))
_LIB = os.path.join(_BUILD, "libapnsearch_fast.so")

os.makedirs(_BUILD, exist_ok=True)
if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
    subprocess.run([os.environ.get("CC", "cc"), "-O3", "-shared", "-fPIC", "-o", _LIB, _SRC, "-lm"],
                   check=True)

_lib = ctypes.CDLL(_LIB)
_i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_i64 = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
_u8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_u64 = np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
_ll, _int, _dbl = ctypes.c_longlong, ctypes.c_int, ctypes.c_double


def _sig(name, args, res=None):
    f = getattr(_lib, name)
    f.argtypes = args
    f.restype = res
    return f


_sig("ddt_cost", [_int, _i32], _ll)
_sig("swap_deltas", [_int, _i32, _i64])
_sig("anneal_swaps", [_int, _i32, _ll, _dbl, _dbl, ctypes.c_uint64, _int, _i32], _ll)
_sig("find_subspaces", [_int, _u8, _int, _i32, _ll, _ll, ctypes.POINTER(_ll)], _ll)
_sig("find_subspaces_pair", [_int, _u8, _int, _i32, _ll, _ll, ctypes.POINTER(_ll), _i64], _ll)
_sig("find_subspaces_part", [_int, _u8, _int, _i32, _ll, _ll, ctypes.POINTER(_ll), _int, _int], _ll)
_sig("find_complementary_pair", [_int, _ll, _i32, _i64], _int)
_sig("switching_kernel", [_int, _i32, _int, _u64, _int], _int)
_sig("quadratic_apn_search", [_int, ctypes.c_uint64, _ll, _int, _i32], _int)
_sig("bilinear_apn_filter", [_int, _ll, _i32, _u8])


def _dim(S):
    n = len(S).bit_length() - 1
    if len(S) != 1 << n:
        raise ValueError("length must be a power of two")
    return n


def ddt_cost(S):
    """sum_{a != 0, b} D(D - 2)/8: 0 iff S is APN; 3 x the number of 2-flats on which S is affine."""
    S = np.ascontiguousarray(S, dtype=np.int32)
    return int(_lib.ddt_cost(_dim(S), S))


def is_apn(S):
    return ddt_cost(S) == 0


def swap_deltas(S):
    """(N, N) array: entry [i, j] (i < j) is the cost change of swapping S[i] and S[j]."""
    S = np.ascontiguousarray(S, dtype=np.int32)
    out = np.zeros(len(S) * len(S), dtype=np.int64)
    _lib.swap_deltas(_dim(S), S, out)
    return out.reshape(len(S), len(S))


def anneal(S, iters, t0, t1, seed, involution=False):
    """Simulated annealing with transpositions (or involution-preserving moves); returns (cost, best)."""
    S = np.array(S, dtype=np.int32)
    best = np.empty_like(S)
    c = _lib.anneal_swaps(_dim(S), S, iters, t0, t1, seed, int(involution), best)
    return int(c), best


def find_subspaces(mask, d, cap=1_000_000, node_limit=10**10):
    """All d-dim linear subspaces inside {w : mask[w]} u {0}, as (count, d) echelon bases.

    Returns None if the cap or the node limit was reached.
    """
    Z = np.ascontiguousarray(mask, dtype=np.uint8)
    out = np.zeros(cap * d, dtype=np.int32)
    nodes = _ll(0)
    k = _lib.find_subspaces(len(Z).bit_length() - 1, Z, d, out, cap, node_limit, ctypes.byref(nodes))
    return None if k < 0 else out[:k * d].reshape(k, d)


def find_subspaces_part(mask, d, part, nparts, cap=1_000_000, node_limit=10**12):
    """One of nparts disjoint slices of find_subspaces (the union over all parts is the full list)."""
    Z = np.ascontiguousarray(mask, dtype=np.uint8)
    out = np.zeros(cap * d, dtype=np.int32)
    nodes = _ll(0)
    k = _lib.find_subspaces_part(len(Z).bit_length() - 1, Z, d, out, cap, node_limit,
                                 ctypes.byref(nodes), part, nparts)
    return None if k < 0 else out[:k * d].reshape(k, d)


def find_subspaces_parallel(mask, d, workers=None, cap=1_000_000):
    """find_subspaces split over threads (the C code releases the GIL)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = workers or os.cpu_count()
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda k: find_subspaces_part(mask, d, k, workers, cap=cap), range(workers)))
    if any(p is None for p in parts):
        return None
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.zeros((0, d), dtype=np.int32)


def find_subspaces_pair(mask, d, cap=1_000_000, node_limit=10**10):
    """Like find_subspaces but stops at the first pair with trivial intersection.

    Returns (status, bases, pair) with status 'pair', 'none' (exhaustive) or 'unknown'.
    """
    Z = np.ascontiguousarray(mask, dtype=np.uint8)
    out = np.zeros(cap * d, dtype=np.int32)
    nodes = _ll(0)
    pair = np.zeros(2, dtype=np.int64)
    r = _lib.find_subspaces_pair(len(Z).bit_length() - 1, Z, d, out, cap, node_limit,
                                 ctypes.byref(nodes), pair)
    if r == -2:
        k = int(pair[1]) + 1
        return "pair", out[:k * d].reshape(k, d), (int(pair[0]), int(pair[1]))
    if r < 0:
        return "unknown", None, None
    return "none", out[:r * d].reshape(r, d), None


def find_complementary_pair(bases):
    """Indices (i, j) of two subspaces with trivial intersection, or None."""
    bases = np.ascontiguousarray(bases, dtype=np.int32)
    pair = np.zeros(2, dtype=np.int64)
    if _lib.find_complementary_pair(bases.shape[1], bases.shape[0], bases, pair):
        return int(pair[0]), int(pair[1])
    return None


def switching_kernel(F, u, max_basis=4096):
    """Basis (dim, N) of {f : F + u f is APN} for an APN function F (Edel-Pott)."""
    F = np.ascontiguousarray(F, dtype=np.int32)
    n = _dim(F)
    N, W = 1 << n, max(1, (1 << n) // 64)
    out = np.zeros(max_basis * W, dtype=np.uint64)
    dim = _lib.switching_kernel(n, F, int(u), out, max_basis)
    if dim < 0:
        raise ValueError("F is not APN or the kernel is too large")
    words = out[:dim * W].reshape(dim, W)
    bits = ((words[:, :, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)
    return bits.reshape(dim, W * 64)[:, :N]


def quadratic_apn_search(n, seed, node_limit=10**6, max_found=1):
    """Randomized DFS for quadratic APN functions on n <= 8 bits; list of tables."""
    N = 1 << n
    out = np.zeros(max_found * N, dtype=np.int32)
    k = _lib.quadratic_apn_search(n, seed, node_limit, max_found, out)
    return [out[i * N:(i + 1) * N].copy() for i in range(k)]


def bilinear_matrix(F, n):
    """v[k][l] = F(e_k + e_l) + F(e_k) + F(e_l) + F(0)."""
    F = np.asarray(F)
    v = np.zeros((n, n), dtype=np.int32)
    for k in range(n):
        for l in range(n):
            if k != l:
                v[k, l] = F[(1 << k) | (1 << l)] ^ F[1 << k] ^ F[1 << l] ^ F[0]
    return v


def bilinear_apn_filter(vs, n):
    """APN test for quadratic functions given by their bilinear matrices (count, n, n)."""
    vs = np.ascontiguousarray(vs, dtype=np.int32).reshape(-1, n * n)
    flags = np.zeros(len(vs), dtype=np.uint8)
    _lib.bilinear_apn_filter(n, len(vs), vs, flags)
    return flags.astype(bool)
