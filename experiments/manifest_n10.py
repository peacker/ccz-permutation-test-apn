"""
Manifest of the 25 classes of quadratic APN functions in dimension 10 (Table 4): for each class
a representative lookup table, its source, the ortho-derivative invariant (differential spectrum
and extended Walsh spectrum of the ortho-derivative), M, what excludes a permutation, and the
sources whose instances were certified EA-equivalent to it (E4, E11). Field: GF(2^10) in the
polynomial basis modulo gf.PRIMITIVE[10], integers as bit vectors. Run after E4 and E11.
Writes data/n10_25_classes.json.
"""
import json
import os

import numpy as np

from common import ROOT
from apnsearch import families
from apnsearch.invariants import ortho_invariant


def main():
    e4 = json.load(open(os.path.join(ROOT, "results", "e04_n10_known_classes.json")))
    e11 = json.load(open(os.path.join(ROOT, "results", "e11_n10_recent_instances.json")))
    reps = dict((d, F) for d, F in json.load(open(os.path.join(ROOT, "data", "n10_known_classes.json"))))
    members = {}
    for src, labels in e11["summary"]["classes_per_source"].items():
        for lab in labels:
            members.setdefault(lab.replace("NEW: ", ""), []).append(src)
    for r in e4["beierle_leander"]:
        if r["family_class"]:
            members.setdefault(r["family_class"], []).append(r["function"])
    out = []
    for r in e4["rows"]:
        F = np.array(reps[r["representative"]])
        out.append({"class": r["representative"], "source": r["family"], "lut": F.tolist(),
                    "ortho_invariant": ortho_invariant(F, 10), "M": r["max_dim_NB"],
                    "excluded_by": r["decided_by"],
                    "certified_members_from_other_sources": sorted(members.get(r["representative"], []))})
    for t in e11["new_classes"]:
        i = int(t["class"].split("#")[1].split()[0])
        F = families.bik_instance_10(i, int(t["class"].split("beta = ")[1]))
        out.append({"class": t["class"], "source": "Budaghyan-Ivkovic-Kaleyski", "lut": F.tolist(),
                    "ortho_invariant": ortho_invariant(F, 10), "M": t["max_dim_NB"],
                    "excluded_by": "dimension bound (M < n/2)" if t["max_dim_NB"] < 5 else "see E11",
                    "certified_members_from_other_sources": sorted(m for m in t["members"]
                                                                  if not m.startswith("Budaghyan-Ivkovic"))})
    assert len(out) == 25 and len({json.dumps(c["ortho_invariant"]) for c in out}) == 25
    with open(os.path.join(ROOT, "data", "n10_25_classes.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(out)} classes -> data/n10_25_classes.json")


if __name__ == "__main__":
    main()
