"""
E3: validation of the dual CCZ test against the exhaustive vector-space search (Section 6).

For each function: number of n-dim Walsh zero spaces from the exhaustive search and from
the dual parametrization (mode 'all'), and the CCZ-to-permutation verdict of both tests.
n = 6: Gold x^3, an APN Kim function (must give an APN permutation), the Edel-Pott
non-quadratic APN function (exhaustive test only; its Walsh supports are not affine).
n = 8: Gold x^3, x^9, x^3 + Tr(x^9), Kasami x^57 (exhaustive only).
"""
import time

import numpy as np

from common import parser, save
from apnsearch import ccz, families, fast
from apnsearch.gf import GF
from apnsearch.invariants import algebraic_degree
from apnsearch.switching import switch_once


def edel_pott_function():
    F = families.edel_pott_source6()
    for u, kd, q, nq, K in switch_once(F, 6):
        if nq:
            from apnsearch.switching import _anf
            high = np.array([bin(m).count("1") >= 3 for m in range(64)])
            for f in K:
                if (_anf(f) * high).any():
                    return u, F ^ (u * f)
    raise RuntimeError("Edel-Pott switching not found")


def row(name, F, n, dual=True):
    t = time.time()
    r = {"function": name, "n": n, "degree": algebraic_degree(F, n)}
    r["exhaustive_count"] = ccz.count_walsh_zero_spaces(F, n)
    st, info = ccz.ccz_to_permutation_exhaustive(F, n)
    r["exhaustive_verdict"] = st
    r["exhaustive_time"] = round(time.time() - t, 2)
    if st == "permutation":
        r["exhaustive_permutation_is_apn"] = info["permutation_is_apn"]
    if dual:
        t = time.time()
        st_all, info_all = ccz.ccz_to_permutation_dual(F, n, mode="all")
        st_p, info_p = ccz.ccz_to_permutation_dual(F, n, mode="pairs")
        r.update({"dual_count": info_all["subspaces"], "dual_verdict": st_p,
                  "dual_time": round(time.time() - t, 2), "max_dim_NB": info_p["max_dim_NB"]})
        if st_p == "permutation":
            r["dual_permutation_is_apn"] = info_p["permutation_is_apn"]
    print(r, flush=True)
    return r


def main():
    parser(__doc__).parse_args()
    g6, g8 = GF(6), GF(8)
    u, kim = families.kim6()
    ue, ep = edel_pott_function()
    rows = [row("x^3", g6.power_table(3), 6), row(f"Kim (u = {u})", kim, 6),
            row(f"Edel-Pott (switching u = {ue})", ep, 6, dual=False),
            row("x^3", g8.power_table(3), 8), row("x^9", g8.power_table(9), 8),
            row("x^3 + Tr(x^9)", families.x3_tr_x9(8), 8),
            row("x^57 (Kasami)", g8.power_table(57), 8, dual=False)]
    save("e03_ccz_validation", {"rows": rows})


if __name__ == "__main__":
    main()
