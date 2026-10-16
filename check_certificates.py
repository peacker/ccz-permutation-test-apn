"""
Independent check of the stored EA-equivalence certificates (E4, E5, E11).

Each certificate (L1, L2) claims that  instance + L1 o reference o L2  is affine, with L1, L2
invertible linear maps given by the images of the basis vectors. This script checks exactly
that, with equivalence.check_certificate, which does not use the search that found the
certificate. It needs only NumPy and the files results/ea_certificates_*.json.

    python3 check_certificates.py
"""
import json
import os

from apnsearch.equivalence import check_certificate

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    total = bad = 0

    def check(ref, inst, n, c, label):
        nonlocal total, bad
        total += 1
        if not check_certificate(ref, inst, n, c["L1"], c["L2"]):
            bad += 1
            print("FAILED:", label)

    d = json.load(open(os.path.join(R, "ea_certificates_e04.json")))
    for c in d["certificates"]:
        check(d["references"][c["reference"]], c["instance"], 10, c, f"E4 {c['reference']}")
    d = json.load(open(os.path.join(R, "ea_certificates_e11.json")))
    for c in d["certificates"]:
        check(c["reference_table"], c["instance_table"], 10, c, f"E11 {c['instance']}")
    d = json.load(open(os.path.join(R, "ea_certificates_e05.json")))
    for n in ("6", "8", "9", "10"):
        for c in d[n]["certificates"]:
            check(d[n]["references"][c["reference"]], c["instance_table"], int(n), c, f"E5 n={n}")
    print(f"{total - bad} of {total} certificates valid")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
