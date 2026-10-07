"""Explicit EA-equivalence of quadratic APN functions, with a checkable certificate.

If G = L1 o F o L2 + A with L1, L2 linear permutations and A affine, the ortho-derivatives satisfy
pi_G = M o pi_F o L2 with M = (L1^T)^-1 (with respect to the dot product of integers as bit
vectors). We enumerate the pairs (L2, M) with this property by a depth-first search over the
images of a basis under L2, pruned with point invariants of the ortho-derivatives (the number of
preimages of pi(a) and the multiset of row a of the DDT of pi), and accept the first pair for
which G + L1 o F o L2 is affine. The search is exhaustive, so `None` means that F and G are
EA-inequivalent; for quadratic APN functions EA- and CCZ-equivalence coincide (Yoshiara).
Every returned pair (L1, L2) is re-checked by `check_certificate`, which does not rely on the
search.
"""
import numpy as np

from .invariants import ortho_derivative


def apply(cols, v):
    """Images of the integer vectors v under the linear map with column images cols."""
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for i, c in enumerate(cols):
        out ^= ((v >> i) & 1) * int(c)
    return out


def rank(cols):
    rows, r = [int(c) for c in cols], 0
    for bit in range(max(rows + [0]).bit_length() - 1, -1, -1):
        p = next((i for i in range(r, len(rows)) if rows[i] >> bit & 1), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i] >> bit & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def is_affine(H, n):
    """H(x + y) + H(x) + H(y) + H(0) = 0 for all x and all basis vectors y."""
    X = np.arange(1 << n)
    return all(np.all(H[X ^ (1 << i)] ^ H ^ H[1 << i] ^ H[0] == 0) for i in range(n))


def check_certificate(F, G, n, L1, L2):
    """True iff L1, L2 are invertible and G + L1 o F o L2 is affine."""
    F, G = np.asarray(F, dtype=np.int64), np.asarray(G, dtype=np.int64)
    if rank(L1) != n or rank(L2) != n:
        return False
    X = np.arange(1 << n)
    return is_affine(G ^ apply(L1, F[apply(L2, X)]), n)


def _point_invariants(pi, n):
    N = 1 << n
    X = np.arange(N)
    pre = np.bincount(pi, minlength=N)[pi]
    out = []
    for a in range(N):
        v, c = np.unique(np.bincount(pi[X ^ a] ^ pi, minlength=N), return_counts=True)
        out.append((int(pre[a]),) + tuple(zip(v.tolist(), c.tolist())))
    return out


class _PartialLinearMap:
    """A linear map over GF(2) known on a subspace, given by reduced (source, image) pairs."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})

    def reduce(self, s, t=0):
        for p in sorted(self.rows, reverse=True):
            if s >> p & 1:
                a, b = self.rows[p]
                s, t = s ^ a, t ^ b
        return s, t

    def add(self, s, t):
        """Impose s -> t; False if this contradicts linearity."""
        s, t = self.reduce(s, t)
        if s == 0:
            return t == 0
        self.rows[s.bit_length() - 1] = (s, t)
        return True


def _transpose_inverse(cols, n):
    """Columns of (M^T)^-1 for the matrix M with column images cols, or None if singular."""
    A = [[(cols[j] >> i) & 1 for j in range(n)] for i in range(n)]       # M[i][j]
    T = [[A[j][i] for j in range(n)] + [int(i == k) for k in range(n)] for i in range(n)]
    r = 0
    for c in range(n):
        p = next((i for i in range(r, n) if T[i][c]), None)
        if p is None:
            return None
        T[r], T[p] = T[p], T[r]
        for i in range(n):
            if i != r and T[i][c]:
                T[i] = [x ^ y for x, y in zip(T[i], T[r])]
        r += 1
    return [sum(T[i][n + j] << i for i in range(n)) for j in range(n)]


def ea_equivalence(F, G, n, node_limit=10**7):
    """(L1, L2) with G + L1 o F o L2 affine, or None if F and G are EA-inequivalent.

    F and G must be quadratic APN functions given by their tables. Raises RuntimeError if the
    node limit is reached.
    """
    F, G = np.asarray(F, dtype=np.int64), np.asarray(G, dtype=np.int64)
    pF, pG = ortho_derivative(F, n), ortho_derivative(G, n)
    iF, iG = _point_invariants(pF, n), _point_invariants(pG, n)
    if sorted(iF[1:]) != sorted(iG[1:]):
        return None
    by_inv = {}
    for a in range(1, 1 << n):
        by_inv.setdefault(iF[a], []).append(a)
    X = np.arange(1 << n)
    nodes = 0

    def dfs(k, L, M):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise RuntimeError("node limit reached")
        if k == n:
            L2 = [L[1 << i] for i in range(n)]
            red = [M.reduce(1 << i) for i in range(n)]
            if any(s for s, _ in red):
                raise RuntimeError("the image of the ortho-derivative does not span F_2^n")
            L1 = _transpose_inverse([t for _, t in red], n)
            if L1 is None or not is_affine(G ^ apply(L1, F[apply(L2, X)]), n):
                return None
            return L1, L2
        b, images = 1 << k, set(L.values())
        for c in by_inv.get(iG[b], []):
            if c in images:
                continue
            M2, new, ok = _PartialLinearMap(M.rows), {}, True
            for x, lx in L.items():
                y, ly = x ^ b, lx ^ c
                if ly == 0 or iF[ly] != iG[y] or not M2.add(int(pF[ly]), int(pG[y])):
                    ok = False
                    break
                new[y] = ly
            if ok:
                r = dfs(k + 1, {**L, **new}, M2)
                if r:
                    return r
        return None

    return dfs(0, {0: 0}, _PartialLinearMap())
