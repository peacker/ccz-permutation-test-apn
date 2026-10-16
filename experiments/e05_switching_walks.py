"""
E5: exact switching closure (Section 5) in dimensions 6, 8, 9 and 10.

Sources: the APN power maps, x^3 + a^-1 Tr(a^3 x^9), and at n = 10 all 25 known quadratic
classes (the 20 of data/n10_known_classes.json, written by e04, and the six instances of
Budaghyan-Ivkovic-Kaleyski). For every class representative and every u the switching kernel is
computed exactly, and every nonzero class of the kernel modulo the trivial part gives one
neighbour (switching.closure). Each quadratic neighbour is either certified EA-equivalent to a
stored representative or becomes a new class; every new class is switched in turn and tested
for CCZ-equivalence to a permutation with the projection test. Non-quadratic APN switchings are
recorded (n = 6 must reproduce the Edel-Pott function).
"""
import json
import os

import numpy as np

from common import ROOT, parser, save
from apnsearch import ccz, families
from apnsearch.gf import GF
from apnsearch.switching import closure


def sources(n):
    src = [(name, F, True) for name, F in families.power_maps(n)]
    src.append(("x^3 + Tr(x^9)", families.x3_tr_x9(n), False))
    if n % 2 == 0:
        g = GF(n)
        beta = int(g.exp[(g.N - 1) // 3])
        src.append(("x^3 + b^-1 Tr(b^3 x^9), b in GF(4)", families.x3_tr_x9(n, beta), False))
    if n == 6:
        src.append(("x^3 + a(x^17 + x^18 + x^20 + x^24)", families.edel_pott_source6(), False))
    if n == 10:
        path = os.path.join(ROOT, "data", "n10_known_classes.json")
        if os.path.exists(path):
            src += [(name, np.array(F), False) for name, F in json.load(open(path))]
        g = GF(10)
        beta = min(x for x in range(2, g.N) if g.pow(x, 4) == x)
        src += [(f"Budaghyan-Ivkovic-Kaleyski #{i}", families.bik_instance_10(i, beta), False)
                for i in families.BIK_TABLE2]
    return src


def main():
    a = parser(__doc__).parse_args()
    out, certs = {}, {}
    for n in (6, 8, 9, 10):
        ccz_results = []

        def on_new(label, G, n=n):
            if n % 2:
                return
            st, info = ccz.ccz_to_permutation_projection(G, n, mode="pairs")
            ccz_results.append({"class": label, "verdict": st, "max_dim_NB": info["max_dim_NB"]})
            print(f"    new class {label}: CCZ to permutation: {st}", flush=True)

        print(f"n = {n}", flush=True)
        res = closure(sources(n), n, on_new_class=on_new)
        certs[str(n)] = {"references": res["reference_tables"], "certificates": res["certificate_list"]}
        out[str(n)] = {"sources": len(sources(n)), "classes_reached": res["class_count"],
                       "distinct_invariants": res["distinct_invariants"],
                       "certificates": res["certificates"],
                       "walk_complete": res["complete"], "nonquadratic": res["nonquadratic"],
                       "new_class_ccz": ccz_results, "stats": res["stats"]}
        print(f"  n = {n}: {res['class_count']} quadratic classes, complete {res['complete']}, "
              f"non-quadratic switchings: {len(res['nonquadratic'])}", flush=True)
    save("e05_switching_walks", out)
    save("ea_certificates_e05", {"meaning": "instance + L1 o reference o L2 is affine", **certs})


if __name__ == "__main__":
    main()
