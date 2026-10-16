"""
E4: CCZ-equivalence to permutations of all known quadratic APN classes in dimension 10.

Instances: Gold x^3, x^9; F4 x^3 + a^-1 Tr(a^3 x^9) for all a in GF(4)* (and a = 1); random
parameter samples of F3 (Budaghyan-Carlet) and F12 (Taniguchi); all F13-type functions
(Budaghyan-Helleseth-Kaleyski) with parameters in GF(4); the five functions of
Beierle-Leander (Zenodo 10.5281/zenodo.4738942). APN-ness is checked for every instance;
classes are told apart by ortho-derivative spectra, and every distinct APN instance is
certified EA-equivalent to the representative of its class (apnsearch/equivalence.py), so the
classes are exact. One representative per class is tested with the projection test (Section 6),
and so is every distinct instance. Also writes data/n10_known_classes.json.
"""
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import ROOT, parser, save
from apnsearch import ccz, equivalence, families, fast
from apnsearch.gf import GF
from apnsearch.invariants import ortho_invariant


def inv_job(F):
    return ortho_invariant(np.array(F), 10)


def condition2_inconclusive(F):
    """True if the 5-dim subspaces of the non-bent components contain a pair spanning F_2^10."""
    import itertools
    W = ccz.walsh_matrix(np.asarray(F), 10)
    rows = ccz.row_structure(W, 10)
    mask = np.zeros(1024, dtype=np.uint8)
    for b in rows:
        mask[b] = 1
    S = fast.find_subspaces(mask, 5)
    spans = []
    for B in (S if S is not None else []):
        sp = {0}
        for v in B:
            sp |= {x ^ int(v) for x in sp}
        spans.append(sp)
    return any(len(a & b) == 1 for a, b in itertools.combinations(spans, 2))


def cert_job(args):
    G, F = (np.array(x) for x in args)
    cert = equivalence.ea_equivalence(G, F, 10)
    return bool(cert and equivalence.check_certificate(G, F, 10, *cert))


def decided_by(M, inconclusive):
    if M < 5:
        return "dimension bound (M < n/2)"
    if not inconclusive:
        return "Proposition 7 (no two candidate projections span F_2^10)"
    return "Theorem 8"


def ccz_job(F):
    st, info = ccz.ccz_to_permutation_projection(np.array(F), 10, mode="pairs")
    return st, info["max_dim_NB"]


def main():
    a = parser(__doc__).parse_args()
    samples = 100 if a.quick else 300
    rng = random.Random(1)
    g = GF(10)
    inst = [("Gold x^3", g.power_table(3)), ("Gold x^9", g.power_table(9))]
    gf4 = [x for x in range(g.N) if g.pow(x, 4) == x and x]
    inst += [(f"F4 a={x}", families.x3_tr_x9(10, x)) for x in gf4 + [2]]
    inst += [families.f3_instance(rng)[::-1] for _ in range(samples)]
    inst += [families.f12_instance(rng)[::-1] for _ in range(samples)]
    inst += [(d, F) for F, d in families.f13_all(10)]
    apn = [(d, F) for d, F in inst if fast.is_apn(F)]
    print(f"{len(inst)} instances, {len(apn)} APN")
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        invs = list(ex.map(inv_job, [F.tolist() for _, F in apn], chunksize=8))
    classes = {}
    for (d, F), inv in zip(apn, invs):
        classes.setdefault(inv, (d.split()[0], d, F))
    bl = families.beierle_leander_10()
    bl_rows = []
    for d, F in bl:
        inv = ortho_invariant(F, 10)
        bl_rows.append({"function": d, "apn": fast.is_apn(F), "in_family_classes": inv in classes,
                        "family_class": classes[inv][1] if inv in classes else None})
        classes.setdefault(inv, ("Beierle-Leander", d, F))
    rows = []
    for fam, d, F in classes.values():
        st, info = ccz.ccz_to_permutation_projection(F, 10, mode="pairs")
        inc = condition2_inconclusive(F)
        rows.append({"family": fam, "representative": d, "verdict": st,
                     "nonbent_rows": info["nonbent_rows"], "max_dim_NB": info["max_dim_NB"],
                     "projections_tested": {str(k): v[0] for k, v in info["per_dim"].items()},
                     "spaces_found": sum(v[1] for v in info["per_dim"].values()),
                     "condition2_inconclusive": inc,
                     "decided_by": decided_by(info["max_dim_NB"], inc)})
        print(rows[-1], flush=True)
    # independent of the invariant: test every distinct APN instance
    distinct = {tuple(F.tolist()): F for _, F in apn}
    distinct.update({tuple(F.tolist()): F for _, F in bl})
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        verdicts = list(ex.map(ccz_job, [F.tolist() for F in distinct.values()], chunksize=4))
    all_instances = {"tested": len(verdicts),
                     "verdicts": {v: sum(1 for x, _ in verdicts if x == v) for v in {x for x, _ in verdicts}},
                     "excluded_by_counting": sum(1 for _, M in verdicts if M < 5)}
    # every distinct instance is EA-equivalent to the representative with the same invariant
    rep_of = {inv: F for inv, (_, _, F) in classes.items()}
    pairs = [(rep_of[ortho_invariant(F, 10)].tolist(), F.tolist()) for F in distinct.values()]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        certified = list(ex.map(cert_job, pairs, chunksize=2))
    all_instances["certified_equivalent_to_class_representative"] = sum(certified)
    print("all distinct APN instances:", all_instances, flush=True)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "n10_known_classes.json"), "w") as f:
        json.dump([(r["representative"], F.tolist()) for r, (_, _, F) in zip(rows, classes.values())], f)
    per_family = {}
    for r in rows:
        per_family[r["family"]] = per_family.get(r["family"], 0) + 1
    print("classes per family:", per_family, "total", len(rows))
    save("e04_n10_known_classes", {"instances": len(inst), "apn_instances": len(apn),
                                   "classes": len(rows), "classes_per_family": per_family,
                                   "beierle_leander": bl_rows, "rows": rows,
                                   "all_instances": all_instances})


if __name__ == "__main__":
    main()
