"""
E11: quadratic APN functions in dimension 10 from sources not covered by the list of E4.

Instances:
  - Zheng, Kan, Li, Peng and Tang (arXiv 2101.11535v1): the three explicit instances of their
    Table III, for every primitive element up to Frobenius, and random members of the family of
    their Theorem 3.3;
  - Budaghyan, Ivkovic and Kaleyski ("Triplicate functions", Table 2): the six instances, for both
    primitive elements of GF(4);
  - Goloğlu's biprojective families F1 and F2 (all q), the two families of Calderini, Li and
    Villa (all parameters), the function of Li, Zhou, Li and Qu (Theorem 5), random members of
    Carlet's family (all on GF(2^5)^2);
  - the representative of x^3 + Tr(x^9) of Yu, Li, Ichanska and Kaleyski.
APN instances are grouped by the ortho-derivative invariant. Every instance is then shown to be
EA-equivalent to a reference function of its group by an explicit, independently checked
certificate (apnsearch/equivalence.py): the representative of the E4 class with the same
invariant, or, for a class not among those of E4, the first instance of the group. New classes
are tested for CCZ-equivalence to a permutation as in E4, and their switching neighbourhoods
are explored as in E5. The two Beierle-Leander functions that E4 identifies with F13-type classes
by the invariant are also certified. Run E4 first.
"""
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import ROOT, parser, save
from apnsearch import ccz, equivalence, families, fast
from apnsearch.gf import GF
from apnsearch.invariants import ortho_invariant
from apnsearch.switching import switch_once, walk
from e04_n10_known_classes import condition2_inconclusive


def certify(args):
    G, F = (np.array(x) for x in args)
    cert = equivalence.ea_equivalence(G, F, 10)
    return bool(cert and equivalence.check_certificate(G, F, 10, *cert))


def inv_job(F):
    return ortho_invariant(np.array(F), 10)


def instances(g, samples):
    rng = random.Random(1)
    out = []
    prim = sorted({min((k << j) % 1023 for j in range(10))
                   for k in range(1, 1023) if math.gcd(k, 1023) == 1})
    for k in prim:
        for d, F in families.zheng_instances_10(int(g.exp[k])):
            out.append(("Zheng et al.", d, f"alpha = g^{k}", F))
    out += [("Zheng et al.", "Thm 3.3 family", d, F) for d, F in families.zheng_family_10(rng, samples)]
    gf4 = [x for x in range(2, g.N) if g.pow(x, 4) == x]
    for i in families.BIK_TABLE2:
        for beta in gf4:
            out.append(("Budaghyan-Ivkovic-Kaleyski", f"Table 2 #{i}", f"beta = {beta}",
                        families.bik_instance_10(i, beta)))
    out += [("Gologlu", d.split(" k=")[0], d, F) for d, F in families.gologlu_10()]
    out += [("Calderini-Li-Villa", d.split(" k=")[0].split(" alpha=")[0], d, F) for d, F in families.clv_10()]
    out += [("Li-Zhou-Li-Qu", d, "", F) for d, F in families.lzlq_10()]
    out += [("Carlet", "Carlet family", d, F) for d, F in families.carlet_10(rng, samples // 4)]
    out.append(("Yu-Li-Ichanska-Kaleyski", "x^513+x^192+x^96+x^48+x^33+x^18+x^12", "",
                families.yu_f2_x3_tr_x9_10()))
    return out


def main():
    a = parser(__doc__).parse_args()
    g = GF(10)
    with open(os.path.join(ROOT, "data", "n10_known_classes.json")) as f:
        known = [(d, np.array(F)) for d, F in json.load(f)]
    seen = {tuple(F.tolist()) for _, F in known}
    known += [(d, F) for d, F in families.beierle_leander_10() if tuple(F.tolist()) not in seen]
    by_inv = {}
    for d, F in known:
        by_inv.setdefault(ortho_invariant(F, 10), []).append((d, F))
    print(f"{len(known)} reference functions, {len(by_inv)} classes", flush=True)
    workers = os.cpu_count()

    # E4 identifies two Beierle-Leander functions with F13-type classes by the invariant only
    pairs = [(m[0], (d, F)) for m in by_inv.values() for d, F in m[1:]]
    with ProcessPoolExecutor(workers) as ex:
        ok = list(ex.map(certify, [(r[1].tolist(), F.tolist()) for r, (_, F) in pairs]))
    e4_pairs = [{"function": d, "class": r[0], "ea_equivalent": o} for (r, (d, _)), o in zip(pairs, ok)]
    print(e4_pairs, flush=True)

    inst = instances(g, 40 if a.quick else 200)
    rows = []
    apn = []
    for src, name, par, F in inst:
        r = {"source": src, "name": name, "parameter": par, "apn": bool(fast.is_apn(F))}
        rows.append(r)
        if r["apn"]:
            apn.append((r, F))
    with ProcessPoolExecutor(workers) as ex:
        invs = list(ex.map(inv_job, [F.tolist() for _, F in apn], chunksize=4))
    new_classes = {}                       # invariant -> (label, reference table)
    jobs = []
    for (r, F), inv in zip(apn, invs):
        if r["source"] == "Budaghyan-Ivkovic-Kaleyski":
            i = int(r["name"].split("#")[1])
            r["spectrum_as_printed"] = [c for _, c in inv[0]] == families.BIK_TABLE2[i][1]
        if inv in by_inv:
            d, G = by_inv[inv][0]
            r["known_class"], r["new_class"] = d, None
        else:
            d, G = new_classes.setdefault(inv, (f"{r['source']} {r['name']} {r['parameter']}".strip(), F))
            r["known_class"], r["new_class"] = None, d
        jobs.append((G.tolist(), F.tolist()))
    t = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for (r, _), o in zip(apn, ex.map(certify, jobs)):
            r["ea_equivalent_to_reference"] = o
    print(f"certificates: {sum(r['ea_equivalent_to_reference'] for r, _ in apn)}/{len(apn)} "
          f"[{time.time() - t:.0f}s]", flush=True)

    tested = []
    for inv, (d, F) in new_classes.items():
        st, info = ccz.ccz_to_permutation_dual(F, 10, mode="pairs")
        sw = switch_once(F, 10, False)                  # exact switching neighbourhood, as in E5
        members = sorted({f"{r['source']}: {r['name']}" for r, _ in apn if r["new_class"] == d})
        tested.append({"class": d, "members": members, "verdict": st, "max_dim_NB": info["max_dim_NB"],
                       "nonbent_rows": info["nonbent_rows"],
                       "condition2_inconclusive": condition2_inconclusive(F),
                       "switching_nontrivial_u": sum(1 for x in sw if x[2] > 0)})
        print(tested[-1], flush=True)

    # switching walk from the new classes (as in E5); the classes reached are compared with E4
    sw_ccz = []

    def on_new(label, G):
        st, info = ccz.ccz_to_permutation_dual(G, 10, mode="pairs")
        sw_ccz.append({"class": label, "verdict": st, "max_dim_NB": info["max_dim_NB"]})

    res = walk([(d, F, False) for d, F in new_classes.values()], 10, seconds=3600,
               samples_per_u=3, on_new_class=on_new)
    reached = [ortho_invariant(np.array(G), 10) for _, G in res["classes"]]
    switching = {"complete": res["complete"], "classes_reached": len(reached),
                 "in_e4_classes": sum(1 for v in reached if v in by_inv),
                 "in_new_classes": sum(1 for v in reached if v in new_classes),
                 "other": sum(1 for v in reached if v not in by_inv and v not in new_classes),
                 "nonquadratic": res["nonquadratic"], "new_class_ccz": sw_ccz}
    print("switching from the new classes:", switching, flush=True)

    per_source = {}
    for r, _ in apn:
        key = f"{r['source']}: {r['name']}"
        per_source.setdefault(key, set()).add(r["known_class"] or "NEW: " + r["new_class"])
    summary = {
        "instances": len(rows), "apn_instances": len(apn),
        "in_e4_classes": sum(1 for r, _ in apn if r["known_class"]),
        "certified": sum(1 for r, _ in apn if r["ea_equivalent_to_reference"]),
        "new_classes": len(new_classes),
        "new_classes_excluded_by_counting": sum(1 for t in tested if t["max_dim_NB"] < 5),
        "classes_per_source": {k: sorted(v) for k, v in sorted(per_source.items())},
        "zheng_example1_apn_parameters": [r["parameter"] for r in rows
                                          if r["name"] == "Zheng et al. Example 1" and r["apn"]],
    }
    print(json.dumps(summary, indent=1))
    save("e11_n10_recent_instances", {"summary": summary, "switching": switching,
                                       "e4_invariant_matches": e4_pairs, "rows": rows,
                                       "new_classes": tested})


if __name__ == "__main__":
    main()
