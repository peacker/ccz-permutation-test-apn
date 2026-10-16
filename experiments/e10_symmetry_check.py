"""
E10 check: the symmetry-reduced subspace search used for n = 22 (apnsearch/symmetry.py) against
the plain search, on the same family x^3 + a^-1 Tr(a^3 x^9) with the same symmetry groups.

For n = 14 and n = 18 and a in {1, alpha, alpha^2}, the subspaces of dimensions m and m+1 (and,
for n = 14, m-2 and m-1, where there are many) of the non-bent components are found (a) by the plain parallel search and (b) by the orbit-pruned search
with the group of output_symmetry(a) (order m for a != 1, order 2m for a = 1); the two lists of
reduced echelon bases must coincide. For n = 18 the plain search takes a few minutes on 128
threads, about an hour on 18 cores.

    python3 e10_symmetry_check.py --n 14 18
"""
import argparse
import os
import time

import numpy as np

from common import save
from apnsearch import ccz, fast, families, symmetry
from apnsearch.gf import GF
from e10_f4_n18 import output_symmetry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[14, 18])
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--extra", type=int, nargs="*", default=[14],
                   help="values of n for which dimensions m-2 and m-1 are compared too")
    args = p.parse_args()
    rows = []
    for n in args.n:
        g, m = GF(n), n // 2
        for a in (1, 2, 4):
            F = families.x3_tr_x9(n, a)
            mask = np.zeros(g.N, dtype=np.uint8)
            mask[list(ccz.row_structure_quadratic(F, n))] = 1
            cols = output_symmetry(g, a)
            for d in ([m - 2, m - 1] if n in args.extra else []) + [m, m + 1]:
                t0 = time.time()
                plain = {symmetry.rref(B) for B in fast.find_subspaces_parallel(mask, d, workers=args.workers)}
                t1 = time.time()
                sym = {tuple(int(v) for v in B)
                       for B in symmetry.find_subspaces_sym(mask, d, cols, workers=args.workers)}
                r = {"n": n, "a": a, "dim": d, "group_order": symmetry.group_order(cols, n),
                     "plain": len(plain), "symmetric": len(sym), "equal": plain == sym,
                     "plain_seconds": round(t1 - t0), "symmetric_seconds": round(time.time() - t1)}
                print(r, flush=True)
                rows.append(r)
    save("e10_symmetry_check", {"rows": rows})


if __name__ == "__main__":
    main()
