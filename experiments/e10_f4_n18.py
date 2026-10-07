"""
E10: x^3 + a^-1 Tr(a^3 x^9) in dimension n = 2m with m odd (default n = 18, m = 9, 3 | m),
for a in {1, alpha, alpha^2}.

Finds all (m+1)- and m-dimensional subspaces of the non-bent components (parallel search),
checks whether two m-dimensional ones span F_2^n (if not, the necessary condition of
Goloğlu-Pavlu already excludes a permutation), and otherwise decides CCZ-equivalence to a
permutation with Theorem 8 (enumeration of all Walsh zero spaces with projection B, for each
m-dimensional B, in parallel processes). Intended for a many-core machine.

    python3 e10_f4_n18.py --workers 128
    python3 e10_f4_n18.py --n 22 --workers 128     # writes results/e10_f4_n22.json
"""
import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save
from apnsearch import ccz, fast, families, symmetry
from apnsearch.gf import GF
from e09_f4_family import dual_basis_map

N_DIM = 18
ROWS = None


def span(B):
    s = {0}
    for v in B:
        s |= {x ^ int(v) for x in s}
    return s


def _init_worker(rows, n):
    global ROWS, N_DIM
    ROWS, N_DIM = rows, n


def output_symmetry(g, a):
    """Column images of B with F_a(A x) = B(F_a(x)): y -> y^2 for a = 1, else y -> a^3 y^4.

    (F_a(x)^4 = F_{a^4}(x^4) and F_{a^4}(y) = a^-3 F_a(a y) give F_a(a x^4) = a^3 F_a(x)^4.)
    The non-bent components b, with b.F(A x) = (B^T b).F(x), are invariant under B^T.
    """
    if a == 1:
        B = [g.pow(1 << i, 2) for i in range(g.n)]
    else:
        a3 = g.pow(a, 3)
        B = [g.mul(a3, g.pow(1 << i, 4)) for i in range(g.n)]
    return symmetry.transpose(B, g.n)


def walsh_spaces_for(Bbasis):
    B = []
    for v in Bbasis:
        nb = ccz.add_to_basis(B, int(v))
        if nb is not None:
            B = nb
    return ccz.enumerate_U(N_DIM, ROWS, B)


def main():
    global ROWS, N_DIM
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--a", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--out", default=None)
    p.add_argument("--sym", action="store_true",
                   help="search subspaces up to the Frobenius-type symmetry (needed for n >= 22)")
    p.add_argument("--dims", default="m,m+1", help="m,m+1 (default) or m: whether to search dimension m+1 too")
    args = p.parse_args()
    N_DIM = args.n
    n, m = N_DIM, N_DIM // 2
    args.out = args.out or f"e10_f4_n{n}"
    g = GF(n)
    out = []
    for a in args.a:
        t0 = time.time()
        F = families.x3_tr_x9(n, a)
        ROWS = ccz.row_structure_quadratic(F, n)
        mask = np.zeros(g.N, dtype=np.uint8)
        for b in ROWS:
            mask[b] = 1
        t_rows = time.time() - t0
        def search(d):
            if args.sym:
                return symmetry.find_subspaces_sym(mask, d, output_symmetry(g, a),
                                                   workers=args.workers, progress=2000)
            return fast.find_subspaces_parallel(mask, d, workers=args.workers)

        dims = args.dims.split(",")
        print(f"a={a}: {len(ROWS)} non-bent components [{time.time() - t0:.0f}s]", flush=True)
        S9 = search(m)
        S10 = search(m + 1) if "m+1" in dims else np.zeros((0, m + 1), dtype=np.int32)
        spans = [span(B) for B in S9]
        spanning_pairs = sum(1 for x, y in itertools.combinations(spans, 2) if len(x & y) == 1)
        r = {"a": a, "log_a_mod_3": int(g.log[a] % 3), "nonbent": len(ROWS),
             "n": n, "subspaces_dim_m_plus_1": len(S10) if "m+1" in dims else None, "subspaces_dim_m": len(S9),
             "spanning_pairs": spanning_pairs, "symmetry": args.sym, "dims": args.dims, "rows_seconds": round(t_rows),
             "search_seconds": round(time.time() - t0 - t_rows)}
        if len(S9):
            L = dual_basis_map(g)
            q = 1 << m
            z = g.pow(2, (g.N - 1) // (q - 1))  # generator of F_q^*
            sub = [0, 1]
            while len(sub) < q:
                sub.append(g.mul(sub[-1], z))
            fields = [{L(w) for w in sp} for sp in spans]
            r["all_spread_lines"] = all(W == {g.mul(next(x for x in W if x), t) for t in sub} for W in fields)
            r["contains_a_line"] = any(W == {g.mul(a, t) for t in sub} for W in fields)
            r["bases_dim_m"] = S9.tolist()
        print({k: v for k, v in r.items() if k != "bases_dim_m"}, flush=True)
        if len(S10) == 0 and spanning_pairs == 0:
            r["verdict"] = "none (necessary condition)"
        else:
            t1 = time.time()
            with ProcessPoolExecutor(min(args.workers, len(S9) + len(S10)),
                                     initializer=_init_worker, initargs=(ROWS, n)) as ex:
                Us = [U for part in ex.map(walsh_spaces_for, list(S9) + list(S10)) for U in part]
            pair = fast.find_complementary_pair(np.array(Us, dtype=np.int32)) if len(Us) >= 2 else None
            r["walsh_zero_spaces"] = len(Us)
            r["verdict"] = "permutation" if pair else "none (Theorem 8)"
            r["theorem8_seconds"] = round(time.time() - t1)
            if pair:
                G = ccz.permutation_from_pair(F, n, Us[pair[0]], Us[pair[1]])
                r["permutation_is_apn"] = bool(fast.is_apn(G))
        print({k: v for k, v in r.items() if k != "bases_dim_m"}, flush=True)
        out.append(r)
        save(args.out, {"rows": out})


if __name__ == "__main__":
    main()
