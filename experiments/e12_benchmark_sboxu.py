"""
E12: CCZ-equivalence to a permutation, our test (Algorithm 1) against sboxU, on the same machine.

Run with SageMath's Python, with sboxU installed (https://github.com/lpp-crypto/sboxU):

    sage -python e12_benchmark_sboxu.py --functions bench_functions.json --timeout 3600

For every function and method, the measurement runs in a fresh subprocess with a time limit.
Methods:
  ours            ccz_to_permutation_projection(F, n, mode="pairs", quadratic=True), one thread
  ours_all        the same in the mode that lists every Walsh zero space (to compare counts)
  sboxu_wzs_1     sboxU.get_WalshZeroesSpaces(F, n_threads=1): all Walsh zero spaces, one thread
  sboxu_wzs_T     the same with T threads (--threads, default 200)
  sboxu_perms     sboxU.enumerate_permutations_in_ccz_class(F) (sboxU's own thread count)
Imports and a warm-up call are excluded from the timing. The output records wall time, the number of Walsh zero spaces where a method lists them, and
whether a permutation exists in the CCZ class.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUNNER = r'''
import json, sys, time
sys.path.insert(0, {root!r})
method, n, threads = {method!r}, {n}, {threads}
F = json.load(open({path!r}))
if method.startswith("ours"):
    import numpy as np
    from apnsearch import ccz
    ccz.ccz_to_permutation_projection(np.array(F), n, mode="pairs", quadratic=True)   # warm-up (C library load)
else:
    import sboxU
t = time.time()                                   # imports and warm-up are not timed
out = {{}}
if method.startswith("ours"):
    st, info = ccz.ccz_to_permutation_projection(np.array(F), n, mode="all" if method == "ours_all" else "pairs",
                                           quadratic=True)
    out["verdict"] = st
    out["max_dim_NB"] = info.get("max_dim_NB")
    if method == "ours_all":
        out["spaces"] = int(sum(v[1] for v in info["per_dim"].values()))
elif method.startswith("sboxu_wzs"):
    from sboxU import get_WalshZeroesSpaces
    w = get_WalshZeroesSpaces(F, n_threads=threads)
    out["spaces"] = len(list(w))
elif method == "sboxu_perms":
    from sboxU import enumerate_permutations_in_ccz_class
    out["permutations"] = len(enumerate_permutations_in_ccz_class(F))
out["seconds"] = round(time.time() - t, 3)
print("RESULT " + json.dumps(out))
'''


def measure(method, fn, threads, timeout, tmpdir):
    path = os.path.join(tmpdir, "F.json")
    json.dump(fn["table"], open(path, "w"))
    code = RUNNER.format(root=ROOT, method=method, n=fn["n"], threads=threads, path=path)
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"timeout": timeout}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r["wall_including_start"] = round(time.time() - t, 3)
            return r
    return {"error": (p.stderr or p.stdout)[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions", required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--threads", type=int, default=200)
    ap.add_argument("--methods", default="ours,ours_all,sboxu_wzs_1,sboxu_wzs_T,sboxu_perms")
    ap.add_argument("--max-n", type=int, default=99)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e12_benchmark_sboxu.json"))
    args = ap.parse_args()
    fns = json.load(open(args.functions))
    tmpdir = os.path.join(ROOT, "results", "bench_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    rows = []
    for fn in sorted((f for f in fns if f["n"] <= args.max_n), key=lambda f: f["n"]):
        row = {"name": fn["name"], "n": fn["n"]}
        for method in args.methods.split(","):
            threads = args.threads if method == "sboxu_wzs_T" else 1
            row[method] = measure(method, fn, threads, args.timeout, tmpdir)
            print(fn["n"], fn["name"], method, row[method], flush=True)
        rows.append(row)
        json.dump({"timeout": args.timeout, "threads": args.threads, "rows": rows}, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
