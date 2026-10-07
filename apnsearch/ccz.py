r"""
CCZ-equivalence to a permutation.

F : F_2^n -> F_2^n is CCZ-equivalent to a permutation iff the Walsh zeros of F,
Z = {(a, b) != 0 : W_F(a, b) = 0} with W_F(a, b) = sum_x (-1)^(a.x + b.F(x)),
contain two n-dimensional subspaces U, V with U n V = {0} [Canteaut-Perrin 2019].
Vectors are packed as w = (a << n) | b.

ccz_to_permutation_exhaustive  generic test: all n-dim subspaces of Z (practical up to n = 8)
ccz_to_permutation_dual        exact test for functions whose Walsh supports
                               S_b = {a : W_F(a, b) != 0} are affine subspaces (e.g. all
                               quadratic functions); see the paper, Section 6:
  U is determined by B = proj_b(U) (dim m), P = U_0^perp (U_0 = U n (F_2^n x 0), dim m) and
  a bilinear map Psi : B x P -> F_2, and U lies in Z u {0} iff for every b in B \ {0}
  some y in P n L_b satisfies Psi(b, y) != y . s_b, where S_b = s_b + L_b^perp.
  A complementary pair needs dim proj(U) + dim proj(V) >= n, hence only projections of
  dimension in [n - M, M] are enumerated (M = largest dimension of a subspace of the
  non-bent rows); if M < n/2 there is no pair at all (counting criterion).
"""

import time

import numpy as np

from . import fast


def walsh_zero_mask(F, n):
    """uint8 mask over w = (a << n) | b with 1 where W_F(a, b) = 0 and w != 0."""
    W = walsh_matrix(F, n)
    mask = (W == 0).astype(np.uint8).reshape(-1)
    mask[0] = 0
    return mask


def permutation_from_pair(F, n, U, V):
    """G = G2 o G1^{-1} with G1(x) = (u_i . (x, F(x)))_i, G2(x) = (v_j . (x, F(x)))_j."""
    N = 1 << n
    graph = (np.arange(N) << n) | np.asarray(F)
    parity = np.array([bin(v).count("1") & 1 for v in range(1 << (2 * n))])
    G1 = sum(parity[graph & u] << i for i, u in enumerate(U))
    G2 = sum(parity[graph & v] << i for i, v in enumerate(V))
    if len(set(G1.tolist())) != N:
        return None
    G = np.zeros(N, dtype=np.int64)
    G[G1] = G2
    return G


def ccz_to_permutation_exhaustive(F, n, node_limit=10**10):
    """Generic exhaustive test. Returns (status, info): 'permutation', 'none' or 'unknown'."""
    t0 = time.time()
    mask = walsh_zero_mask(F, n)
    status, bases, pair = fast.find_subspaces_pair(mask, n, node_limit=node_limit)
    info = {"walsh_zeros": int(mask.sum()), "time": time.time() - t0}
    if status == "unknown":
        return "unknown", info
    if status == "none":
        info["subspaces"] = len(bases)
        return "none", info
    G = permutation_from_pair(F, n, bases[pair[0]].tolist(), bases[pair[1]].tolist())
    info["permutation"] = G.tolist()
    info["permutation_is_apn"] = fast.is_apn(G)
    return "permutation", info


def count_walsh_zero_spaces(F, n, node_limit=10**10):
    """Number of n-dim subspaces of Z u {0} (exhaustive; for validation)."""
    bases = fast.find_subspaces(walsh_zero_mask(F, n), n, node_limit=node_limit)
    return None if bases is None else len(bases)


# --------------------------------------------------------------------------- GF(2) helpers

def reduce_vec(v, basis):
    """Reduce v by an echelon basis (distinct leading bits, sorted descending)."""
    for r in basis:
        v = min(v, v ^ r)
    return v


def add_to_basis(basis, v):
    """Return a new echelon basis spanning basis + v (or None if v is dependent)."""
    v = reduce_vec(v, basis)
    if v == 0:
        return None
    new = [min(r, r ^ v) for r in basis] + [v]
    return sorted(new, reverse=True)


def span_key(basis):
    return tuple(basis)


def span_elements(basis):
    s = [0]
    for v in basis:
        s += [x ^ v for x in s]
    return s


def dot(x, y):
    return bin(x & y).count("1") & 1


def nullspace(rows, n):
    """Basis of {y in F_2^n : y . r = 0 for all r in rows}."""
    piv = {}
    for r in rows:
        for p, pr in piv.items():
            if r >> p & 1:
                r ^= pr
        if r:
            p = r.bit_length() - 1
            for q in list(piv):
                if piv[q] >> p & 1:
                    piv[q] ^= r
            piv[p] = r
    free = [i for i in range(n) if i not in piv]
    out = []
    for f in free:
        y = 1 << f
        for p, pr in piv.items():
            if pr >> f & 1:
                y |= 1 << p
        out.append(y)
    return out


def dual_basis(P, n):
    """Vectors d_k with P[j] . d_k = delta_jk."""
    m = len(P)
    out = []
    for k in range(m):
        others = [P[j] for j in range(m) if j != k]
        cand = nullspace(others, n)
        d = next(c for c in cand if dot(c, P[k]))
        out.append(d)
    return out


# --------------------------------------------------------------------------- Walsh structure

def walsh_matrix(F, n):
    """W[a, b] = sum_x (-1)^(a.x + b.F(x))."""
    N = 1 << n
    X = np.arange(N)
    bits = np.array([bin(v).count("1") & 1 for v in range(N)], dtype=np.int64)
    W = 1 - 2 * bits[np.asarray(F)[:, None] & X[None, :]]
    h = 1
    while h < N:
        W = W.reshape(N // (2 * h), 2, h, N)
        W = np.concatenate([W[:, 0] + W[:, 1], W[:, 0] - W[:, 1]], axis=1).reshape(N, N)
        h *= 2
    return W


def row_structure_quadratic(F, n):
    """Same output as row_structure(walsh_matrix(F, n), n) for a quadratic F, computed by
    linear algebra instead of the Walsh transform: L_b is the radical of the alternating
    form b . B(x, y), and s_b is any solution of s . y = F_b(y) + F_b(0) for y in L_b."""
    F = np.asarray(F)
    N = 1 << n
    from .invariants import algebraic_degree
    if algebraic_degree(F, n) > 2:
        raise NotAffineSupports("row_structure_quadratic needs a quadratic function")
    v = fast.bilinear_matrix(F, n).astype(np.int64)        # v[k][l] = B(e_k, e_l)
    par = np.array([bin(x).count("1") & 1 for x in range(N)], dtype=np.int64)
    rows = {}
    for b in range(1, N):
        M = par[v & b]                                      # n x n bit matrix of b . B
        R = [int("".join(map(str, M[k][::-1])), 2) for k in range(n)]
        L = nullspace(R, n)                                 # radical (M is symmetric)
        if not L:
            continue                                        # bent component
        f0 = par[int(F[0]) & b]
        # solve s . y = F_b(y) + F_b(0) for all y in the basis of L
        targets = [int(par[int(F[y]) & b] ^ f0) for y in L]
        s = _solve_dot(L, targets, n)
        rows[b] = (s, L)
    return rows


def _solve_dot(Y, t, n):
    """Some s in F_2^n with s . Y[i] = t[i] for all i (Y linearly independent)."""
    piv = []                                                # (pivot bit, row, target)
    for y, c in zip(Y, t):
        for p, r, rc in piv:
            if y >> p & 1:
                y ^= r
                c ^= rc
        p = y.bit_length() - 1
        piv = [(q, r ^ y if r >> p & 1 else r, rc ^ c if r >> p & 1 else rc) for q, r, rc in piv]
        piv.append((p, y, c))
    s = 0
    for p, r, c in piv:
        # in reduced form each row has a distinct pivot bit not present in the others
        if c:
            s |= 1 << p
    return s


class NotAffineSupports(Exception):
    pass


def row_structure(W, n):
    """For each non-bent row b != 0: (s_b, basis of L_b). Raises if a support is not affine."""
    N = 1 << n
    rows = {}
    for b in range(1, N):
        S = np.nonzero(W[:, b])[0]
        if len(S) == N:
            continue
        s = int(S[0])
        D = [int(a) ^ s for a in S]
        size = len(D)
        if size & (size - 1):
            raise NotAffineSupports(f"row {b}: support size {size}")
        basis = []
        for v in D:
            nb = add_to_basis(basis, v)
            if nb is not None:
                basis = nb
        if 1 << len(basis) != size:
            raise NotAffineSupports(f"row {b}: support is not an affine subspace")
        rows[b] = (s, nullspace(basis, n))
    return rows


# --------------------------------------------------------------------------- enumeration

def enumerate_P(Bnz, rows, m, n, cap):
    """All m-dim subspaces P of F_2^n meeting L_b nontrivially for every b in Bnz."""
    L = {b: span_elements(rows[b][1])[1:] for b in Bnz}
    found = set()

    def hits(basis, b):
        return any(reduce_vec(y, basis) == 0 for y in L[b])

    def complete(basis):
        # all m-dim subspaces containing span(basis), deduplicated by echelon key
        if len(basis) == m:
            found.add(span_key(basis))
            if len(found) > cap:
                raise RuntimeError("too many P")
            return
        for v in range(1, 1 << n):
            nb = add_to_basis(basis, v)
            if nb is not None and span_key(nb) not in seen_partial:
                seen_partial.add(span_key(nb))
                complete(nb)

    seen_partial = set()

    def rec(basis):
        pending = next((b for b in Bnz if not hits(basis, b)), None)
        if pending is None:
            complete(basis)
            return
        if len(basis) == m:
            return
        for y in L[pending]:
            nb = add_to_basis(basis, y)
            if nb is not None and span_key(nb) not in seen_partial:
                seen_partial.add(span_key(nb))
                rec(nb)

    rec([])
    return [list(k) for k in found]


def enumerate_U(n, rows, B, cap=10**6):
    """All n-dim subspaces U of Z u {0} with projection exactly B (echelon basis of B)."""
    m = len(B)
    if m == 0:
        return [[(1 << i) << n for i in range(n)]]      # the a-axis
    Belems = span_elements(B)                        # index t -> b = sum t_i B[i]
    Bnz = Belems[1:]
    out = []
    for P in enumerate_P(Bnz, rows, m, n, cap):
        # coordinates of vectors of P in the basis P (P is echelon; solve by reduction)
        Pel = span_elements(P)
        coord = {v: t for t, v in enumerate(Pel)}
        conds = []  # per b (index t): list of (coord mask of y, y . s_b)
        for t in range(1, 1 << m):
            b = Belems[t]
            s, Lb = rows[b]
            Lset = set(span_elements(Lb)[1:])
            I = [(coord[y], dot(y, s)) for y in Pel[1:] if y in Lset]
            conds.append((t, I))
        sols = _solve_psi(conds, m)
        if sols is None:                                  # fall back to the plain search
            # DFS over psi_i (m-bit masks on P coordinates)
            by_top = {}
            for t, I in conds:
                by_top.setdefault(t.bit_length() - 1, []).append((t, I))
            psis = [0] * m
            sols = []

            def ok(t, I):
                val = 0
                for i in range(m):
                    if t >> i & 1:
                        val ^= psis[i]
                return any((bin(val & c).count("1") & 1) != sb for c, sb in I)

            def rec(i):
                if i == m:
                    sols.append(list(psis))
                    return
                for psi in range(1 << m):
                    psis[i] = psi
                    if all(ok(t, I) for t, I in by_top.get(i, [])):
                        rec(i + 1)

            rec(0)
        if not sols:
            continue
        U0 = nullspace(P, n)
        d = dual_basis(P, n)
        for psi_list in sols:
            basis = [(u << n) for u in U0]
            for i in range(m):
                phi = 0
                for k in range(m):
                    if psi_list[i] >> k & 1:
                        phi ^= d[k]
                basis.append((phi << n) | B[i])
            out.append(basis)
            if len(out) > cap:
                raise RuntimeError("too many U")
    return out


def _solve_psi(conds, m, max_free=22):
    """All bilinear maps Psi : B x P -> F_2 (as lists of m-bit masks psi_i) satisfying
    condition (3) of Theorem 8, by linear algebra.

    For b with P n L_b = {0, y} the condition is the linear equation
    Psi(b, y) = 1 + y . s_b in the m*m entries x_{i,k} of Psi; b with a larger intersection
    give 'not all equal' constraints, checked on the solution space afterwards. Returns None
    if that space is too large to enumerate (the caller then uses the plain search).
    """
    def form(t, c):
        f = 0
        for i in range(m):
            if t >> i & 1:
                for k in range(m):
                    if c >> k & 1:
                        f |= 1 << (i * m + k)
        return f

    eqs, deferred = [], []
    for t, I in conds:
        if not I:
            return []                                   # P misses L_b: no Walsh zero space
        if len(I) == 1:
            c, sb = I[0]
            eqs.append((form(t, c), sb ^ 1))
        else:
            deferred.append([(form(t, c), sb) for c, sb in I])
    piv = {}                                            # pivot bit -> (row, rhs)
    for f, r in eqs:
        for p_, (pf, pr) in piv.items():
            if f >> p_ & 1:
                f ^= pf
                r ^= pr
        if f == 0:
            if r:
                return []                               # inconsistent
            continue
        p_ = f.bit_length() - 1
        for q in list(piv):
            qf, qr = piv[q]
            if qf >> p_ & 1:
                piv[q] = (qf ^ f, qr ^ r)
        piv[p_] = (f, r)
    nv = m * m
    free = [v for v in range(nv) if v not in piv]
    if deferred and len(free) > max_free:
        return None
    x0 = 0
    for p_, (pf, pr) in piv.items():
        if pr:
            x0 |= 1 << p_
    kernel = []
    for fv in free:
        k = 1 << fv
        for p_, (pf, pr) in piv.items():
            if pf >> fv & 1:
                k |= 1 << p_
        kernel.append(k)
    if len(kernel) > max_free:
        return None
    par = lambda v: bin(v).count("1") & 1
    sols = []
    for mask in range(1 << len(kernel)):
        x = x0
        for j, k in enumerate(kernel):
            if mask >> j & 1:
                x ^= k
        if all(any(par(x & f) != sb for f, sb in D) for D in deferred):
            sols.append([(x >> (i * m)) & ((1 << m) - 1) for i in range(m)])
    return sols


def ccz_to_permutation_dual(F, n, mode="pairs", node_limit=10**10, verbose=False,
                            quadratic=False):
    """Returns (status, info); status 'permutation' or 'none'.

    mode 'pairs' only enumerates U whose projection can take part in a
    complementary pair; mode 'all' enumerates every n-dim subspace (for testing).
    """
    t0 = time.time()
    N = 1 << n
    F = np.asarray(F)
    if quadratic:
        rows = row_structure_quadratic(F, n)
    else:
        rows = row_structure(walsh_matrix(F, n), n)
    nb_mask = np.zeros(N, dtype=np.uint8)
    for b in rows:
        nb_mask[b] = 1
    # largest subspace dimension inside the non-bent rows; in mode 'pairs' only the
    # dimensions n - M <= d <= M are enumerated
    M = 0
    by_dim = {}
    for d in range(n, 0, -1):
        if M and mode != "all" and d < n - M:
            break
        bases = fast.find_subspaces(nb_mask, d, cap=5_000_000, node_limit=node_limit)
        if bases is None:
            raise RuntimeError(f"subspace enumeration in NB (dim {d}) hit its limit")
        if len(bases):
            by_dim[d] = bases
            M = max(M, d)
    lo = 0 if mode == "all" else max(0, n - M)
    all_U = []
    per_dim = {}
    for m in range(lo, M + 1):
        Bs = [[]] if m == 0 else by_dim.get(m, np.zeros((0, m), dtype=np.int32)).tolist()
        cnt = 0
        for Bb in Bs:
            B = []
            for v in Bb:
                B = add_to_basis(B, v)
            Us = enumerate_U(n, rows, B)
            cnt += len(Us)
            all_U += Us
        per_dim[m] = (len(Bs), cnt)
        if verbose:
            print(f"    dim B = {m}: {len(Bs)} projections, {cnt} subspaces "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    info = {"nonbent_rows": len(rows), "max_dim_NB": M, "per_dim": per_dim,
            "subspaces": len(all_U)}
    pair = None
    if len(all_U) >= 2:
        pair = fast.find_complementary_pair(np.array(all_U, dtype=np.int32))
    info["time"] = time.time() - t0
    if pair is None:
        return "none", info
    G = permutation_from_pair(F, n, all_U[pair[0]], all_U[pair[1]])
    assert G is not None and sorted(G.tolist()) == list(range(N))
    info["permutation"] = G.tolist()
    info["permutation_is_apn"] = fast.is_apn(G)
    return "permutation", info
