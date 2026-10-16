"""
E10, n = 18, a = 1: where the three 9-dimensional candidate projections fail in Theorem 8.

For each candidate B (results/e10_f4_n18_a1.json) we list the dimensions of the spaces L_b,
b in B \\ {0}, the admissible subspaces P (enumerate_P: P meets every L_b nontrivially), the sizes
of P n L_b, and the number of bilinear maps Psi satisfying condition (3) of Theorem 8.
"""
import json
import os
import time

from common import ROOT, save
from apnsearch import ccz, families
from apnsearch.ccz import _solve_psi, add_to_basis, dot, enumerate_P, span_elements


def main():
    n, m = 18, 9
    rows = ccz.row_structure_quadratic(families.x3_tr_x9(n, 1), n)
    bases = json.load(open(os.path.join(ROOT, "results", "e10_f4_n18_a1.json")))["rows"][0]["bases_dim_9"]
    out = []
    for Bb in bases:
        t0 = time.time()
        B = []
        for v in Bb:
            B = add_to_basis(B, v)
        Belems = span_elements(B)
        Ps = enumerate_P(Belems[1:], rows, m, n, 10**6)
        per_P = []
        for P in Ps:
            Pel = span_elements(P)
            coord = {v: i for i, v in enumerate(Pel)}
            conds = []
            for t in range(1, 1 << m):
                s, Lb = rows[Belems[t]]
                Lset = set(span_elements(Lb)[1:])
                conds.append((t, [(coord[y], dot(y, s)) for y in Pel[1:] if y in Lset]))
            sols = _solve_psi(conds, m)
            per_P.append({"P": [int(v) for v in P],
                          "nonzero_in_P_cap_Lb": sorted({len(I) for _, I in conds}),
                          "psi_solutions": len(sols)})
        r = {"B": Bb, "dim_Lb": sorted({len(rows[b][1]) for b in Belems[1:]}),
             "admissible_P": len(Ps), "per_P": per_P, "seconds": round(time.time() - t0, 1)}
        print({k: v for k, v in r.items() if k != "per_P"},
              [(p["nonzero_in_P_cap_Lb"], p["psi_solutions"]) for p in per_P], flush=True)
        out.append(r)
    save("e10_n18_failure_stage", {"rows": out})


if __name__ == "__main__":
    main()
