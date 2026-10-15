"""
E7: sampling symmetry-restricted families of univariate quadratic functions in dimension 10
(Section 7.4). F(x) = sum_e c_e x^e, e = 2^i + 2^j, with
  gf2       c_e in GF(2) (Frobenius-invariant), density p
  subfield  c_e in GF(2^k), k | 10, sparse support
  sparse    2-5 random terms
  residue   exponents e = d (mod m), m | 1023 (F(zeta x) = zeta^d F(x), zeta of order m)
APN-ness from the bilinear matrix (fast.bilinear_apn_filter); classes from ortho-derivative
spectra, compared with the 20 known classes (data/n10_known_classes.json from e04); every
new class is tested with the projection test.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import ROOT, parser, save
from apnsearch import ccz, fast
from apnsearch.gf import GF
from apnsearch.invariants import ortho_invariant


def setup(n):
    g = GF(n)
    exps = sorted({(1 << i) + (1 << j) for i in range(n) for j in range(i + 1, n)})
    basis = [1 << k for k in range(n)]
    V = {}
    for e in exps:
        M = np.zeros((n, n), dtype=np.int64)
        for k in range(n):
            for l in range(n):
                if k != l:
                    x, y = basis[k], basis[l]
                    M[k, l] = (g.pow(x ^ y, e) ^ g.pow(x, e) ^ g.pow(y, e))
        V[e] = M
    return g, exps, V


def families(n):
    q1 = (1 << n) - 1
    fams = [("gf2", p) for p in (0.1, 0.2, 0.35, 0.5)]
    fams += [("subfield", k) for k in range(2, n) if n % k == 0]
    fams += [("sparse", t) for t in (2, 3, 4, 5)]
    for m in sorted({m for m in range(3, q1) if q1 % m == 0}):
        if m > 400:
            continue
        for d in range(m):
            fams.append(("residue", (m, d)))
    return fams


def exponents_for(fam, exps):
    kind, par = fam
    if kind == "residue":
        m, d = par
        return [e for e in exps if e % m == d]
    return exps


def worker(args):
    n, fam, seconds, seed = args
    g, exps, V = setup(n)
    N = g.N
    rng = np.random.default_rng(seed)
    kind, par = fam
    E = exponents_for(fam, exps)
    if not E:
        return fam, 0, []
    Vs = np.array([V[e] for e in E])                      # (len(E), n, n)
    sub = None
    if kind == "subfield":
        sub = np.array([x for x in range(N) if g.pow(x, 1 << par) == x])
    t0 = time.time()
    tried, hits = 0, []
    batch = 2000
    while time.time() - t0 < seconds and len(hits) < 2000:
        if kind == "gf2":
            C = (rng.random((batch, len(E))) < par).astype(np.int64)
        elif kind == "subfield":
            C = sub[rng.integers(0, len(sub), (batch, len(E)))]
            C *= rng.random((batch, len(E))) < 0.25
        elif kind == "sparse":
            C = np.zeros((batch, len(E)), dtype=np.int64)
            for r in range(batch):
                idx = rng.choice(len(E), min(par, len(E)), replace=False)
                C[r, idx] = rng.integers(1, N, len(idx))
        else:
            C = rng.integers(0, N, (batch, len(E)))
        # v = sum_e c_e V_e over GF(2^n)
        v = np.zeros((batch, n, n), dtype=np.int64)
        for j in range(len(E)):
            v ^= g.mul_vec(np.repeat(C[:, j], n * n).reshape(batch, n, n),
                           np.broadcast_to(Vs[j], (batch, n, n)))
        flags = fast.bilinear_apn_filter(v, n)
        tried += batch
        for r in np.nonzero(flags)[0]:
            hits.append([int(c) for c in C[r]])
    return fam, tried, [(E, c) for c in hits]


def table(g, E, coefs):
    F = np.zeros(g.N, dtype=np.int64)
    X = np.arange(g.N)
    for e, c in zip(E, coefs):
        if c:
            F ^= g.mul_vec(np.full(g.N, c), g.pow_vec(X, e))
    return F



def inv_job(args):
    n, i, F = args
    return i, ortho_invariant(np.array(F), n)


def main():
    a = parser(__doc__).parse_args()
    n = 10
    seconds = 10 if a.quick else 60
    per_family = 50 if a.quick else 150
    g, exps, V = setup(n)
    known = {ortho_invariant(np.array(F), n): name
             for name, F in json.load(open(os.path.join(ROOT, "data", "n10_known_classes.json")))}
    fams = families(n)
    t0 = time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        results = list(ex.map(worker, [(n, fam, seconds, s) for s, fam in enumerate(fams)]))
    tried = sum(r[1] for r in results)
    todo, seen, cnt = [], set(), {}
    for fam, _, hs in results:
        for E, c in hs:
            key = (tuple(E), tuple(c))
            if key in seen or cnt.get(fam, 0) >= per_family:
                continue
            seen.add(key)
            cnt[fam] = cnt.get(fam, 0) + 1
            todo.append((fam, E, c))
    apn_total = sum(len(r[2]) for r in results)
    print(f"{len(fams)} families, {tried} candidates, {apn_total} APN; classifying {len(todo)}",
          flush=True)
    tables = [(n, i, table(g, E, c).tolist()) for i, (fam, E, c) in enumerate(todo)]
    new = {}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for i, inv in ex.map(inv_job, tables, chunksize=8):
            if inv not in known:
                new.setdefault(inv, i)
    rows = []
    for inv, i in new.items():
        fam, E, c = todo[i]
        st, info = ccz.ccz_to_permutation_projection(np.array(tables[i][2]), n, mode="pairs")
        rows.append({"family": [str(x) for x in fam], "terms": [(e, cc) for e, cc in zip(E, c) if cc],
                     "verdict": st, "max_dim_NB": info["max_dim_NB"]})
        print("NEW class:", rows[-1], flush=True)
    by_kind = {}
    for fam, tr, hs in results:
        k = fam[0]
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += tr
        by_kind[k][1] += len(hs)
    save("e07_n10_sampling", {"families": len(fams), "candidates": tried, "apn": apn_total,
                              "classified": len(todo), "new_classes": rows,
                              "by_kind": by_kind, "seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
