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
