"""
E10, n = 18, a = 1: where the three 9-dimensional candidate projections fail in Theorem 8.

For each candidate B (results/e10_f4_n18_a1.json) we list the dimensions of the spaces L_b,
b in B \\ {0}, the admissible subspaces P (enumerate_P: P meets every L_b nontrivially), the sizes
of P n L_b, and the number of bilinear maps Psi satisfying condition (3) of Theorem 8.

When every P n L_b is {0, y_b}, condition (3) is the linear system Psi(b, y_b) = 1 + y_b . s_b,
b in B - {0}, in the m*m entries of Psi. For an inconsistent system we store a witness: a set of
equations whose left-hand sides sum to zero and whose right-hand sides sum to 1. Each equation
is stored as (t, y, s), where b = sum_i t_i B_i, y is in P n L_b, and s = y . s_b; anyone can
recompute the left-hand sides from B and P and check the witness (verify.py does).
"""
import json
import os
import time

from common import ROOT, save
from apnsearch import ccz, families
from apnsearch.ccz import _solve_psi, add_to_basis, dot, enumerate_P, span_elements


def form(t, c, m):
    """Bit mask over the m*m entries Psi(B_i, p_k) of Psi(b, y), b = sum t_i B_i, y = sum c_k p_k."""
    f = 0
    for i in range(m):
        if t >> i & 1:
            for k in range(m):
                if c >> k & 1:
                    f |= 1 << (i * m + k)
    return f


def witness(rows):
    """rows: list of (form, rhs). Indices of rows summing to (0, 1), or None if consistent."""
    piv = {}                                   # pivot bit -> (form, rhs, set of row indices)
    for j, (f, r) in enumerate(rows):
        comb = 1 << j
        for p_, (pf, pr, pc) in piv.items():
            if f >> p_ & 1:
                f, r, comb = f ^ pf, r ^ pr, comb ^ pc
        if f == 0:
            if r:
                return [i for i in range(len(rows)) if comb >> i & 1]
            continue
        p_ = f.bit_length() - 1
        for q in list(piv):
            qf, qr, qc = piv[q]
            if qf >> p_ & 1:
                piv[q] = (qf ^ f, qr ^ r, qc ^ comb)
        piv[p_] = (f, r, comb)
    return None


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
            rec = {"P": [int(v) for v in P], "nonzero_in_P_cap_Lb": sorted({len(I) for _, I in conds}),
                   "psi_solutions": len(sols)}
            if all(len(I) == 1 for _, I in conds):
                eqs = [(form(t, I[0][0], m), I[0][1] ^ 1) for t, I in conds]
                w = witness(eqs)
                if w is not None:
                    rec["witness"] = [[conds[j][0], int(Pel[conds[j][1][0][0]]), conds[j][1][0][1]] for j in w]
            per_P.append(rec)
        r = {"B": Bb, "dim_Lb": sorted({len(rows[b][1]) for b in Belems[1:]}),
             "admissible_P": len(Ps), "per_P": per_P, "seconds": round(time.time() - t0, 1)}
        print({k: v for k, v in r.items() if k != "per_P"},
              [(p["nonzero_in_P_cap_Lb"], p["psi_solutions"]) for p in per_P], flush=True)
        out.append(r)
    save("e10_n18_failure_stage", {"rows": out})


if __name__ == "__main__":
    main()
