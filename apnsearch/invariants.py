"""
Invariants used to tell CCZ-classes apart.

For a quadratic APN function F the ortho-derivative pi(a) is the nonzero vector
orthogonal to the hyperplane {B(a, x) : x}, B(a, x) = F(x + a) + F(x) + F(a) + F(0).
Its differential spectrum and extended Walsh spectrum are EA-invariants of F, hence
CCZ-invariants for quadratic APN functions [Yoshiara 2012; Kaleyski 2022].
"""

import numpy as np

from . import fast


def _fwht(M):
    N = M.shape[0]
    h = 1
    while h < N:
        M = M.reshape(N // (2 * h), 2, h, -1)
        M = np.concatenate([M[:, 0] + M[:, 1], M[:, 0] - M[:, 1]], axis=1).reshape(N, -1)
        h *= 2
    return M


def ortho_derivative(F, n):
    N = 1 << n
    v = fast.bilinear_matrix(F, n).astype(np.int64)
    A = np.arange(N)
    R = np.zeros((N, n), dtype=np.int64)                  # R[a, i] = B(a, e_i)
    for k in range(n):
        R ^= np.where(((A >> k) & 1)[:, None] == 1, v[k][None, :], 0)
    parity = np.array([bin(x).count("1") & 1 for x in range(N)], dtype=np.uint8)
    C = np.arange(1, N)
    pi = np.zeros(N, dtype=np.int64)
    for a in range(1, N):
        idx = np.nonzero(~parity[C[:, None] & R[a][None, :]].any(axis=1))[0]
        if len(idx) != 1:
            raise ValueError("not a quadratic APN function")
        pi[a] = C[idx[0]]
    return pi


def ortho_invariant(F, n):
    """(differential spectrum, extended Walsh spectrum) of the ortho-derivative."""
    N = 1 << n
    pi = ortho_derivative(F, n)
    X = np.arange(N)
    ddt = {}
    for a in range(1, N):
        vals, cnts = np.unique(np.bincount(pi[X ^ a] ^ pi, minlength=N), return_counts=True)
        for v, c in zip(vals, cnts):
            ddt[int(v)] = ddt.get(int(v), 0) + int(c)
    parity = np.array([bin(x).count("1") & 1 for x in range(N)], dtype=np.int64)
    W = _fwht(1 - 2 * parity[pi[:, None] & X[None, :]])
    walsh = np.unique(np.abs(W[:, 1:]), return_counts=True)
    return (tuple(sorted(ddt.items())), tuple((int(a), int(b)) for a, b in zip(*walsh)))


def algebraic_degree(F, n):
    deg = 0
    for bit in range(n):
        a = (np.asarray(F) >> bit & 1).astype(np.uint8)
        for i in range(n):
            a = a.reshape(-1, 2, 1 << i)
            a[:, 1, :] ^= a[:, 0, :]
            a = a.reshape(-1)
        idx = np.nonzero(a)[0]
        if len(idx):
            deg = max(deg, max(bin(int(m)).count("1") for m in idx))
    return deg
