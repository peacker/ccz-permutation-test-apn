"""Subspace search up to a linear symmetry group of the mask.

If a linear map T of F_2^n preserves Z = {w : mask[w]}, the d-dimensional subspaces of
Z u {0} are found up to the group G = <T> as follows. Let r_1 < r_2 < ... be the smallest
elements of the G-orbits in Z. Every such subspace V has a G-image that contains some r_i
and meets no orbit of r_j with j < i (take the first orbit that V meets). Search i therefore
looks for the d-dimensional subspaces that contain r_i inside Z minus those earlier orbits:
in the quotient F_2^n / <r_i> these are the (d-1)-dimensional subspaces of
{w : w and w + r_i both in Z}. The searches are independent and run in parallel processes;
their results are closed under G and deduplicated.

For F_a(x) = x^3 + a^-1 Tr(a^3 x^9) with a = alpha^k one has F_a(alpha^k x^4) = alpha^{3k} F_a(x)^4
(and F_1(x^2) = F_1(x)^2), so the non-bent components are invariant under the transpose of
y -> alpha^{3k} y^4 (order n/2) or, for a = 1, of y -> y^2 (order n).
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import fast

_MASK = _OMIN = None


def apply_linear(cols, v):
    """Images of the integer vectors v under the linear map with column images cols."""
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for i, c in enumerate(cols):
        out ^= ((v >> i) & 1) * int(c)
    return out


def transpose(cols, n):
    return [sum(((int(cols[j]) >> i) & 1) << j for j in range(n)) for i in range(n)]


def group_order(cols, n, limit=10**4):
    basis = np.array([1 << i for i in range(n)], dtype=np.int64)
    v = basis.copy()
    for k in range(1, limit):
        v = apply_linear(cols, v)
        if np.array_equal(v, basis):
            return k
    raise ValueError("map is not invertible or has a large order")


def orbit_minima(cols, n):
    """For every vector w, the smallest element of its orbit under <T>."""
    w = np.arange(1 << n, dtype=np.int64)
    omin, cur = w.copy(), w
    for _ in range(group_order(cols, n) - 1):
        cur = apply_linear(cols, cur)
        np.minimum(omin, cur, out=omin)
    return omin


def rref(vectors):
    """Reduced echelon basis of the span (tuple, decreasing pivots)."""
    rows = []
    for v in vectors:
        v = int(v)
        for r in rows:
            v = min(v, v ^ r)
        if v:
            rows = [min(r, r ^ v) for r in rows] + [v]
    return tuple(sorted(rows, reverse=True))


def _init(mask, omin):
    global _MASK, _OMIN
    _MASK, _OMIN = mask, omin


def _search_from(args):
    r, d, n = args
    p = r.bit_length() - 1
    low = (1 << p) - 1
    q = np.arange(1 << (n - 1), dtype=np.int64)
    w = (q & low) | ((q & ~low) << 1)            # vectors with bit p cleared
    live = _MASK.astype(bool) & (_OMIN >= r)       # Z minus the orbits before that of r
    sub = (live[w] & live[w ^ r]).astype(np.uint8)
    sub[0] = 0
    S = fast.find_subspaces(sub, d - 1, node_limit=10**18)
    if S is None:
        raise RuntimeError(f"cap reached for r = {r}")
    expand = lambda x: (x & low) | ((x & ~low) << 1)
    return [rref([r] + [expand(int(x)) for x in B]) for B in S]


def find_subspaces_sym(mask, d, cols, workers=None, progress=None):
    """All d-dim subspaces of {w : mask[w]} u {0} for a mask invariant under the map cols.

    Returns an array (count, d) of reduced echelon bases, like fast.find_subspaces.
    """
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    n = len(mask).bit_length() - 1
    w = np.flatnonzero(mask)
    img = apply_linear(cols, w)
    if not mask[img].all():
        raise ValueError("the mask is not invariant under the given map")
    omin = orbit_minima(cols, n)
    reps = [int(r) for r in np.flatnonzero(mask.astype(bool) & (omin == np.arange(1 << n)))]
    G = [list(cols)]
    for _ in range(group_order(cols, n) - 2):
        G.append([int(x) for x in apply_linear(cols, np.array(G[-1]))])
    found = set()
    with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init,
                             initargs=(mask, omin)) as ex:
        for k, part in enumerate(ex.map(_search_from, [(r, d, n) for r in reps], chunksize=4)):
            for B in part:
                found.add(B)
                for g in G:
                    found.add(rref(apply_linear(g, np.array(B))))
            if progress and k % progress == 0:
                print(f"  [sym] {k}/{len(reps)} representatives, {len(found)} subspaces", flush=True)
    out = sorted(found)
    return np.array(out, dtype=np.int32).reshape(len(out), d)
