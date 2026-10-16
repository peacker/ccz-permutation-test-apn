"""
Exact switching (Edel-Pott 2009). For an APN function F and u != 0, G = F + u f is APN iff
f(x) + f(x+a) + f(y) + f(y+a) = 0 for all a != 0 and x, y with
F(x) + F(x+a) + F(y) + F(y+a) = u: a linear system in the truth table of f, solved by
fast.switching_kernel. The trivial solutions are the affine functions and the coordinate
functions of F; solutions outside that space give switching neighbours. For quadratic F,
a solution whose algebraic normal form has a monomial of degree >= 3 gives a non-quadratic
APN function.

walk() explores the switching graph breadth-first: every new quadratic class (told apart
by ortho-derivative spectra) is queued and switched in turn.
"""

import os
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import fast
from .gf import GF
from .invariants import algebraic_degree, ortho_invariant


def _anf(f):
    a = np.array(f, dtype=np.uint8) & 1
    n = len(a).bit_length() - 1
    for i in range(n):
        a = a.reshape(-1, 2, 1 << i)
        a[:, 1, :] ^= a[:, 0, :]
        a = a.reshape(-1)
    return a


def _to_int(v):
    return int("".join(map(str, np.asarray(v, dtype=np.uint8)[::-1])), 2)


def _rank(vs):
    basis = []
    for v in vs:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return len(basis)


def trivial_space(F, n):
    N = 1 << n
    X = np.arange(N)
    rows = [np.ones(N, dtype=np.uint8)] + [((X >> i) & 1).astype(np.uint8) for i in range(n)]
    rows += [((np.asarray(F) >> i) & 1).astype(np.uint8) for i in range(n)]
    return [_to_int(r) for r in rows]


def nonquadratic_part_rank(K, n):
    """Rank of the degree >= 3 ANF parts of the kernel basis K."""
    high = np.array([bin(m).count("1") >= 3 for m in range(1 << n)])
    return _rank([_to_int(_anf(f) * high) for f in K])


def u_representatives(F, n, is_power):
    """All u != 0, or representatives modulo x -> cx for a power map x^d (gcd(d, 2^n - 1) of them)."""
    g = GF(n)
    if not is_power:
        return list(range(1, g.N))
    d = int(g.log[int(F[2])])
    import math
    return [int(g.exp[i]) for i in range(math.gcd(d, g.N - 1))]


def switch_once(F, n, is_power=False, threads=None):
    """For every u: (u, kernel dim, nontrivial dim, non-quadratic rank, kernel basis)."""
    us = u_representatives(F, n, is_power)
    with ThreadPoolExecutor(threads or os.cpu_count()) as ex:
        kernels = list(ex.map(lambda u: fast.switching_kernel(F, u), us))
    T = trivial_space(F, n)
    rT = _rank(T)
    quadratic = algebraic_degree(F, n) <= 2
    out = []
    for u, K in zip(us, kernels):
        q = _rank(T + [_to_int(f) for f in K]) - rT
        nq = nonquadratic_part_rank(K, n) if (q and quadratic) else 0
        out.append((u, len(K), q, nq, K))
    return out


def walk(sources, n, seconds=3600, samples_per_u=4, seed=0, on_new_class=None, log=print):
    """Breadth-first walk over the switching graph from (name, table, is_power) sources.

    Returns dict with the reached quadratic classes, non-quadratic finds and per-source stats.
    """
    rng = np.random.default_rng(seed)
    classes, queue, nonquad, stats = {}, [], [], []
    for name, F, is_power in sources:
        F = np.asarray(F)
        if algebraic_degree(F, n) == 2:
            inv = ortho_invariant(F, n)
            if inv in classes:
                continue
            classes[inv] = name
        queue.append((name, F, is_power))
    t0 = time.time()
    qi = 0
    while qi < len(queue) and time.time() - t0 < seconds:
        name, F, is_power = queue[qi]
        qi += 1
        res = switch_once(F, n, is_power)
        nontrivial = [r for r in res if r[2] > 0]
        for u, kd, q, nq, K in nontrivial:
            if nq:
                nonquad.append({"source": name, "u": u, "nonquadratic_rank": nq})
            if algebraic_degree(F, n) > 2:
                continue
            for _ in range(samples_per_u):
                f = (rng.integers(0, 2, len(K)) @ K) % 2
                G = F ^ (u * f)
                if algebraic_degree(G, n) != 2:
                    continue
                inv = ortho_invariant(G, n)
                if inv not in classes:
                    label = f"switch[{len(classes)}]"
                    classes[inv] = label
                    queue.append((label, G, False))
                    if on_new_class:
                        on_new_class(label, G)
        stats.append({"source": name, "nontrivial_u": len(nontrivial),
                      "kernel_dims": sorted({r[1] for r in res})})
        log(f"  {name}: {len(nontrivial)} u with nontrivial switchings; "
            f"classes {len(classes)}, queue {len(queue) - qi}")
    return {"classes": [(lab, F.tolist()) for lab, F, _ in queue],
            "class_count": len(classes), "nonquadratic": nonquad, "stats": stats,
            "complete": qi == len(queue)}


def safe_trivial_space(F, n, u):
    """Span of the affine functions and of the components b.F with b.u = 0.

    For f in this space F + u f is EA-equivalent to F: affine f adds an affine function, and for
    b.u = 0 we have F + u (b.F) = L o F with the invertible map L(y) = y + u (b.y). Moreover, if
    G = F + u f then G + u t, for t in this space, is EA-equivalent to G (b.F = b.G when b.u = 0).
    """
    N = 1 << n
    X = np.arange(N)
    Fa = np.asarray(F, dtype=np.int64)
    rows = [np.ones(N, dtype=np.uint8)] + [((X >> i) & 1).astype(np.uint8) for i in range(n)]
    basis = []
    for b in range(1, N):                      # a basis of u^perp
        if bin(b & u).count("1") % 2:
            continue
        v = b
        for c in basis:
            v = min(v, v ^ c)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
        if len(basis) == n - 1:
            break
    for b in basis:
        y = Fa & b
        par = np.zeros(N, dtype=np.uint8)
        for i in range(n):
            par ^= ((y >> i) & 1).astype(np.uint8)
        rows.append(par)
    return [_to_int(r) for r in rows]


def quotient_basis(K, T):
    """Elements of K (as 0/1 arrays) whose classes form a basis of span(K) modulo span(T)."""
    basis, out = [], []
    def reduce(v):
        for b in basis:
            v = min(v, v ^ b)
        return v
    for t in T:
        v = reduce(t)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    for f in K:
        v = reduce(_to_int(f))
        if v:
            basis.append(v)
            basis.sort(reverse=True)
            out.append(np.asarray(f, dtype=np.uint8))
    return out


def closure(sources, n, on_new_class=None, log=print, threads=None):
    """Exact closure of the quadratic APN classes under switching, with certified equivalences.

    For every class representative F and every u (modulo x -> cx for power maps) the kernel
    K(F, u) is computed, and every nonzero class of K(F, u) modulo safe_trivial_space(F, n, u)
    gives one function G = F + u f (other members of the same class give functions
    EA-equivalent to G). A quadratic G is compared with the stored representatives of equal
    ortho-derivative invariant by the exhaustive EA-equivalence search of equivalence.py; a
    returned certificate is re-checked independently, and G becomes a new representative
    (and is switched in turn) only if it is EA-inequivalent to all of them. Since
    EA-equivalent functions have EA-equivalent switching neighbours, switching one
    representative per class suffices, and the result is the exact set of quadratic classes
    reachable from the sources. Non-quadratic switchings are recorded but not switched further.
    """
    from . import equivalence
    reps, by_inv, nonquad, stats, certs = [], {}, [], [], []
    certificates = 0

    def lookup(G):
        """Index of the representative EA-equivalent to G, or None; certificates are checked."""
        nonlocal certificates
        for i in by_inv.get(ortho_invariant(G, n), []):
            cert = equivalence.ea_equivalence(reps[i][1], G, n)
            if cert is not None:
                assert equivalence.check_certificate(reps[i][1], G, n, *cert)
                certificates += 1
                certs.append({"reference": reps[i][0], "instance_table": np.asarray(G).tolist(),
                              "L1": [int(c) for c in cert[0]], "L2": [int(c) for c in cert[1]]})
                return i
        return None

    def add(name, F, is_power):
        reps.append((name, F, is_power))
        by_inv.setdefault(ortho_invariant(F, n), []).append(len(reps) - 1)

    for name, F, is_power in sources:
        F = np.asarray(F)
        if algebraic_degree(F, n) != 2:
            reps.append((name, F, is_power))          # switched, but not a quadratic class
            continue
        if lookup(F) is None:
            add(name, F, is_power)
    qi = 0
    while qi < len(reps):
        name, F, is_power = reps[qi]
        qi += 1
        res = switch_once(F, n, is_power, threads)
        quadratic = algebraic_degree(F, n) == 2
        nontrivial, quotient_dims, functions = 0, [], 0
        for u, kd, q, nq, K in res:
            if not q:
                continue
            Q = quotient_basis(K, safe_trivial_space(F, n, u))
            if not Q:
                continue
            nontrivial += 1
            quotient_dims.append(len(Q))
            if nq:
                nonquad.append({"source": name, "u": int(u), "nonquadratic_rank": nq})
            if not quadratic:
                continue
            for c in range(1, 1 << len(Q)):
                f = np.zeros_like(Q[0])
                for j in range(len(Q)):
                    if c >> j & 1:
                        f ^= Q[j]
                G = np.asarray(F) ^ (int(u) * f.astype(np.int64))
                functions += 1
                if algebraic_degree(G, n) != 2:
                    continue
                if lookup(G) is None:
                    label = f"switch[{len(reps)}]"
                    add(label, G, False)
                    if on_new_class:
                        on_new_class(label, G)
        stats.append({"source": name, "nontrivial_u": nontrivial,
                      "quotient_dims": dict(sorted(Counter(quotient_dims).items())),
                      "neighbours_examined": functions})
        log(f"  {name}: {nontrivial} u with nontrivial switchings, quotient dims "
            f"{stats[-1]['quotient_dims']}; classes {len(by_inv)}, queue {len(reps) - qi}")
    return {"classes": [(lab, F.tolist()) for lab, F, _ in reps if algebraic_degree(F, n) == 2],
            "class_count": sum(len(v) for v in by_inv.values()),
            "distinct_invariants": len(by_inv), "certificates": certificates,
            "certificate_list": certs,
            "reference_tables": {lab: np.asarray(F).tolist() for lab, F, _ in reps},
            "nonquadratic": nonquad, "stats": stats, "complete": True}
