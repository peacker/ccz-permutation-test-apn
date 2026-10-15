"""
E8: the recursive tree search of Beierle and Leander for quadratic APN functions with a
prescribed linear self-equivalence, in dimension 10 (Section 7.5).

    python3 e08_tree_search.py build                 # clone github.com/cbe90/quadratic_apn, compile
    python3 e08_tree_search.py sweep --seconds 300   # deterministic search, every class
    python3 e08_tree_search.py campaign --workers 128 --hours 24 --slice 1800
                                                     # randomized search on the unfinished classes
    python3 e08_tree_search.py collect               # verify, classify, CCZ-test all solutions

The code of Beierle and Leander (GPL-3.0, Zenodo 10.5281/zenodo.4305864) is not
redistributed; `build` fetches it into external/. Solutions are written to
results/e08_raw/; `collect` compares their classes with data/n10_known_classes.json (from
e04) and writes results/e08_tree_search.json.
"""
import argparse
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from common import RESULTS, ROOT, save
from apnsearch import ccz, fast
from apnsearch.invariants import ortho_invariant

REPO = "https://github.com/cbe90/quadratic_apn.git"
LE = os.path.join(ROOT, "external", "quadratic_apn", "10bit", "LE_automorphisms")
RAW = os.path.join(RESULTS, "e08_raw")
STATE = os.path.join(RESULTS, "e08_sweep.json")


def classes():
    return sorted((d for d in os.listdir(LE) if d.startswith("class")), key=lambda c: int(c[5:]))


def build(_):
    ext = os.path.join(ROOT, "external")
    os.makedirs(ext, exist_ok=True)
    if not os.path.isdir(os.path.join(ext, "quadratic_apn")):
        subprocess.run(["git", "clone", "--depth", "1", REPO], cwd=ext, check=True)
    for c in classes():
        d = os.path.join(LE, c)
        subprocess.run(["cc", "-O3", "-w", "-o", "quad_apn", "quad_apn.c"], cwd=d, check=True)
        subprocess.run(["c++", "-O3", "-w", "-o", "quad_apn_rand", "quad_apn_rand.c"], cwd=d,
                       check=True)
    print(f"built {len(classes())} classes")


def sweep(a):
    os.makedirs(RAW, exist_ok=True)

    def run(c):
        out = os.path.join(RAW, f"{c}_det.txt")
        try:
            subprocess.run([os.path.join(LE, c, "quad_apn"), out], cwd=os.path.join(LE, c),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=a.seconds)
            finished = True
        except subprocess.TimeoutExpired:
            finished = False
        sols = open(out).read().count("{0x") if os.path.exists(out) else 0
        print(f"{c}: finished={finished} solutions={sols}", flush=True)
        return c, finished, sols

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, classes()))
    json.dump({"seconds": a.seconds, "classes": [{"class": c, "finished": f, "solutions": s}
                                                 for c, f, s in res]}, open(STATE, "w"), indent=1)
    print(f"finished {sum(f for _, f, _ in res)} / {len(res)}; "
          f"with solutions: {[c for c, _, s in res if s]}")


def campaign(a):
    os.makedirs(RAW, exist_ok=True)
    todo = [r["class"] for r in json.load(open(STATE))["classes"] if not r["finished"]]
    deadline = time.time() + a.hours * 3600
    order, counter, lock = itertools.cycle(todo), itertools.count(), threading.Lock()

    def worker(w):
        while time.time() < deadline and not os.path.exists(os.path.join(ROOT, "STOP")):
            with lock:
                c, k = next(order), next(counter)
            secs = int(min(a.slice, deadline - time.time()))
            if secs < 30:
                return
            subprocess.run(["nice", "-n", "10", os.path.join(LE, c, "quad_apn_rand"),
                            os.path.join(RAW, f"{c}_rand{k}.txt"), str(secs)],
                           cwd=os.path.join(LE, c), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

    print(f"{len(todo)} classes, {a.workers} workers, {a.hours} h", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(worker, range(a.workers)))


def _inv(lut):
    return ortho_invariant(np.array(lut), 10)


def _ccz(lut):
    st, info = ccz.ccz_to_permutation_projection(np.array(lut), 10, mode="pairs")
    return st


def collect(_):
    known = {ortho_invariant(np.array(F), 10): name
             for name, F in json.load(open(os.path.join(ROOT, "data", "n10_known_classes.json")))}
    tables = {}
    for f in glob.glob(os.path.join(RAW, "*.txt")):
        c = os.path.basename(f).split("_")[0]
        for block in re.findall(r"\{([^}]*)\}", open(f).read()):
            lut = tuple(int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", block))
            if len(lut) == 1024:
                tables.setdefault(lut, c)
    luts = list(tables)
    apn = [l for l in luts if fast.is_apn(l)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        invs = list(ex.map(_inv, apn, chunksize=8))
    found, new = {}, []
    for lut, inv in zip(apn, invs):
        if inv in found:
            continue
        found[inv] = tables[lut]
        if inv not in known:
            st, info = ccz.ccz_to_permutation_projection(np.array(lut), 10, mode="pairs")
            new.append({"le_class": tables[lut], "verdict": st, "max_dim_NB": info["max_dim_NB"],
                        "table": list(lut)})
            print(f"NEW class from {tables[lut]}: CCZ to permutation {st}", flush=True)
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        verdicts = list(ex.map(_ccz, apn, chunksize=4))
    all_verdicts = {v: verdicts.count(v) for v in set(verdicts)}
    print("projection test on every solution:", all_verdicts, flush=True)
    sweep_state = json.load(open(STATE)) if os.path.exists(STATE) else None
    save("e08_tree_search", {
        "sweep": None if sweep_state is None else {
            "seconds": sweep_state["seconds"], "classes": len(sweep_state["classes"]),
            "finished": sum(r["finished"] for r in sweep_state["classes"]),
            "with_solutions": [r["class"] for r in sweep_state["classes"] if r["solutions"]]},
        "solutions": len(luts), "apn": len(apn), "classes": len(found),
        "known_classes_hit": sorted({known[i] for i in found if i in known}),
        "le_classes_with_solutions": sorted(set(tables.values()), key=lambda c: int(c[5:])),
        "new_classes": new, "ccz_all_solutions": all_verdicts})
    print(f"{len(luts)} solutions, {len(apn)} APN, {len(found)} classes, {len(new)} new")


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("sweep")
    s.add_argument("--seconds", type=int, default=300)
    s.add_argument("--workers", type=int, default=os.cpu_count())
    c = sub.add_parser("campaign")
    c.add_argument("--workers", type=int, default=os.cpu_count())
    c.add_argument("--hours", type=float, default=24)
    c.add_argument("--slice", type=int, default=1800)
    sub.add_parser("collect")
    a = p.parse_args()
    {"build": build, "sweep": sweep, "campaign": campaign, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
