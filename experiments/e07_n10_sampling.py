"""
E7: sampling symmetry-restricted families of univariate quadratic functions in dimension 10
(Section 7.4). F(x) = sum_e c_e x^e, e = 2^i + 2^j, with
  gf2       c_e in GF(2) (Frobenius-invariant), density p
  subfield  c_e in GF(2^k), k | 10, sparse support
  sparse    2-5 random terms
  residue   exponents e = d (mod m), m | 1023 (F(zeta x) = zeta^d F(x), zeta of order m)
APN-ness from the bilinear matrix (fast.bilinear_apn_filter). Each family gets a maximum budget
of BATCHES batches of 2000 candidates (seed = index of the family). Before each batch the family
stops if it has already found 2000 APN functions (raw hits, with repetitions), so it can end
with slightly more; all hits are kept. A residue family whose class contains no quadratic
exponent has no candidates. results/e07_n10_sampling.json records, for every family, the seed,
the budget, the batches executed, the APN hits before the last batch, the candidates, the raw
APN hits, the distinct APN functions, the stopping reason and the number of functions tested.
results/e07_functions.csv.gz lists every distinct APN function (families that produced it,
nonzero terms e:c of the polynomial, projection-test verdict, class whose invariant it has). Every distinct APN function kept is tested individually with the projection
test, and its ortho-derivative spectra are compared with the 20 known classes
(data/n10_known_classes.json from e04).
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


BATCHES = 2560


def worker(args):
    n, fam, batches, seed = args
    g, exps, V = setup(n)
    N = g.N
    rng = np.random.default_rng(seed)
    kind, par = fam
    E = exponents_for(fam, exps)
    if not E:
        return fam, 0, [], {"exponents": 0, "executed_batches": 0, "stop": "no exponents"}
    Vs = np.array([V[e] for e in E])                      # (len(E), n, n)
    sub = None
    if kind == "subfield":
        sub = np.array([x for x in range(N) if g.pow(x, 1 << par) == x])
    tried, hits, done, stop, before_last = 0, [], 0, "budget", 0
    batch = 2000
    for _ in range(batches):
        if len(hits) >= 2000:
            stop = "APN cap"
            break
        done += 1
        before_last = len(hits)
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
    if stop == "budget" and len(hits) >= 2000:
        stop = "budget (APN cap reached in the final batch)"
    return fam, tried, [(E, c) for c in hits], {"exponents": len(E), "executed_batches": done, "stop": stop,
                                                "apn_hits_before_last_batch": before_last}


def table(g, E, coefs):
    F = np.zeros(g.N, dtype=np.int64)
    X = np.arange(g.N)
    for e, c in zip(E, coefs):
        if c:
            F ^= g.mul_vec(np.full(g.N, c), g.pow_vec(X, e))
    return F



def test_job(args):
    """Ortho-derivative invariant and projection-test verdict of one function."""
    n, i, E, c = args
    F = table(GF(n), E, c)
    st, info = ccz.ccz_to_permutation_projection(F, n, mode="pairs", quadratic=True)
    return i, ortho_invariant(F, n), st, info["max_dim_NB"]


def main():
    p = parser(__doc__)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    n = 10
    batches = BATCHES // 10 if a.quick else BATCHES
    g, exps, V = setup(n)
    known = {ortho_invariant(np.array(F), n): name
             for name, F in json.load(open(os.path.join(ROOT, "data", "n10_known_classes.json")))}
    fams = families(n)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(worker, [(n, fam, batches, s) for s, fam in enumerate(fams)]))
    tried = sum(r[1] for r in results)
    todo, index = [], {}
    for fam, _, hs, _ in results:
        for E, c in hs:
            key = (tuple(E), tuple(c))
            if key not in index:
                index[key] = len(todo)
                todo.append((fam, E, c))
    apn_total = sum(len(r[2]) for r in results)
    print(f"{len(fams)} families, {tried} candidates, {apn_total} APN, {len(todo)} distinct; testing all",
          flush=True)
    verdicts, classes, new, M = {}, {}, {}, {}
    verdict_of, class_of = {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, inv, st, m in ex.map(test_job, [(n, i, E, c) for i, (_, E, c) in enumerate(todo)],
                                    chunksize=16):
            verdict_of[i] = st
            class_of[i] = known.get(inv, "new")
            verdicts[st] = verdicts.get(st, 0) + 1
            M[m] = M.get(m, 0) + 1
            if inv in known:
                classes[known[inv]] = classes.get(known[inv], 0) + 1
            else:
                new.setdefault(inv, (i, st, m))
    per_family = []
    for s_, (fam, tr, hs, info) in enumerate(results):
        keys = {index[(tuple(E), tuple(c))] for E, c in hs}
        per_family.append({"family": [str(x) for x in fam], "seed": s_, "max_batches": batches,
                           "batch_size": 2000, **info, "candidates": tr, "apn_hits": len(hs),
                           "distinct_apn": len(keys), "tested": sum(1 for k in keys if k in verdict_of),
                           "permutation_found": sum(1 for k in keys if verdict_of.get(k) == "permutation")})
    # one line per distinct function: index, families (seeds) that produced it, nonzero terms
    # e:c of the polynomial, projection-test verdict, class whose invariant it has
    fams_of = {}
    for s_, (fam, tr, hs, info) in enumerate(results):
        for E, c in hs:
            fams_of.setdefault(index[(tuple(E), tuple(c))], set()).add(s_)
    import gzip
    import io
    path = os.path.join(ROOT, "results", "e07_functions.csv.gz")
    with io.TextIOWrapper(gzip.GzipFile(path, "wb", mtime=0)) as f:     # mtime=0: reproducible file
        f.write("index,families,terms,verdict,invariant_of_class\n")
        for i, (_, E, c) in enumerate(todo):
            terms = " ".join(f"{e}:{cc}" for e, cc in zip(E, c) if cc)
            f.write(f"{i},{';'.join(map(str, sorted(fams_of[i])))},{terms},{verdict_of[i]},{class_of[i]}\n")
    rows = []
    for inv, (i, st, m) in new.items():
        fam, E, c = todo[i]
        rows.append({"family": [str(x) for x in fam], "terms": [(e, cc) for e, cc in zip(E, c) if cc],
                     "verdict": st, "max_dim_NB": m})
        print("NEW class:", rows[-1], flush=True)
    by_kind = {}
    for fam, tr, hs, _ in results:
        k = fam[0]
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += tr
        by_kind[k][1] += len(hs)
    save("e07_n10_sampling", {"families": len(fams), "batches_per_family": batches,
                              "candidates": tried, "apn": apn_total, "distinct": len(todo),
                              "tested": sum(verdicts.values()), "verdicts": verdicts,
                              "max_dim_NB": {str(k): v for k, v in sorted(M.items())},
                              "classes": classes, "new_classes": rows,
                              "by_kind": by_kind, "per_family": per_family,
                              "seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
