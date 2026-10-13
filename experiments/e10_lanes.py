"""
E10, distributed version: the subspace search of e10_f4_n18.py --sym, split into lanes that can
run on several machines, with checkpoints.

The orbit representatives r_0 < r_1 < ... of the symmetry group (apnsearch/symmetry.py) are
independent searches. Lane j consists of the indices i = j mod L, processed in decreasing order;
every finished index is appended to DIR/lane_<j>.jsonl, so a lane can be stopped and resumed,
also on another machine (copy the file). A run of e10_f4_n18.py --sym processes the indices in
increasing order and logs "[sym] k/N representatives, c subspaces" (indices 0..k, c subspaces
after closing under the group); `merge` combines such a log line with the lanes once the lanes
cover every index above k.

    python3 e10_lanes.py prepare --n 22 --a 2 --dir D     # non-bent components -> D/mask.npy
    python3 e10_lanes.py run --dir D --lanes 0-5 --workers 256
    python3 e10_lanes.py status --dir D
    python3 e10_lanes.py merge --dir D --ascending K C    # K, C from the ascending run's log
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import common  # noqa: F401  (puts the artifact root on sys.path)
from apnsearch import ccz, families, symmetry
from apnsearch.gf import GF
from e09_f4_family import dual_basis_map
from e10_f4_n18 import output_symmetry

NLANES = 12


def setup(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    n, a = meta["n"], meta["a"]
    mask = np.load(os.path.join(d, "mask.npy"))
    cols = output_symmetry(GF(n), a)
    omin = symmetry.orbit_minima(cols, n)
    reps = np.flatnonzero(mask.astype(bool) & (omin == np.arange(1 << n)))
    return meta, mask, cols, omin, reps


def done_indices(d):
    out = {}
    for f in glob.glob(os.path.join(d, "lane_*.jsonl")):
        for line in open(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:            # a line cut by a crash
                continue
            out[rec["i"]] = rec["S"]
    return out


def frontier(done, nreps):
    """Smallest F such that every index >= F is done."""
    F = nreps
    while F > 0 and (F - 1) in done:
        F -= 1
    return F


def cmd_prepare(args):
    os.makedirs(args.dir, exist_ok=True)
    t = time.time()
    F = families.x3_tr_x9(args.n, args.a)
    rows = ccz.row_structure_quadratic(F, args.n)
    mask = np.zeros(1 << args.n, dtype=np.uint8)
    for b in rows:
        mask[b] = 1
    np.save(os.path.join(args.dir, "mask.npy"), mask)
    json.dump({"n": args.n, "a": args.a, "m": args.n // 2, "nonbent": int(mask.sum())},
              open(os.path.join(args.dir, "meta.json"), "w"))
    print(f"{int(mask.sum())} non-bent components [{time.time() - t:.0f}s]", flush=True)


def lane_set(spec):
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def cmd_run(args):
    meta, mask, cols, omin, reps = setup(args.dir)
    n, m = meta["n"], meta["m"]
    done = done_indices(args.dir)
    lanes = lane_set(args.lanes)
    if args.indices:                                  # explicit list, in the given order (helper runs)
        todo = [int(x) for x in open(args.indices).read().split() if int(x) not in done]
        lanes = sorted({i % NLANES for i in todo})
    else:
        todo = [i for i in range(len(reps) - 1, -1, -1) if i % NLANES in lanes and i not in done]
    print(f"{len(reps)} representatives; lanes {lanes}: {len(todo)} to do", flush=True)
    files = {j: open(os.path.join(args.dir, f"lane_{j}{args.tag}.jsonl"), "a") for j in lanes}
    with ProcessPoolExecutor(args.workers, initializer=symmetry._init, initargs=(mask, omin)) as ex:
        futs = {ex.submit(symmetry._search_from, (int(reps[i]), m, n)): i for i in todo}
        for k, fut in enumerate(as_completed(futs)):
            i = futs[fut]
            f = files[i % NLANES]
            f.write(json.dumps({"i": i, "r": int(reps[i]), "S": [list(B) for B in fut.result()]}) + "\n")
            f.flush()
            if k % 500 == 0:
                print(f"  {time.strftime('%m-%d %H:%M')} {k + 1}/{len(todo)} done", flush=True)


def cmd_status(args):
    meta, mask, cols, omin, reps = setup(args.dir)
    done = done_indices(args.dir)
    F = frontier(done, len(reps))
    found = sum(1 for S in done.values() if S)
    print(json.dumps({"representatives": len(reps), "done": len(done), "frontier": F,
                      "indices_with_subspaces": found}))


def spread_lines_in_mask(meta, mask):
    """Field-coordinate lines c F_q (c up to F_q^*) whose bit-coordinate image lies in the mask."""
    n, m = meta["n"], meta["m"]
    g, q = GF(n), 1 << meta["m"]
    L = dual_basis_map(g)                                  # bit coordinates -> field coordinates
    Lcols = [L(1 << i) for i in range(n)]
    rows = [(Lcols[i], 1 << i) for i in range(n)]           # (field, bit)
    piv = {}
    for fv, bv in rows:
        for p in sorted(piv, reverse=True):
            if fv >> p & 1:
                fv ^= piv[p][0]
                bv ^= piv[p][1]
        piv[fv.bit_length() - 1] = (fv, bv)

    def Linv(x):
        b = 0
        for p in sorted(piv, reverse=True):
            if x >> p & 1:
                x ^= piv[p][0]
                b ^= piv[p][1]
        return b

    z = g.pow(2, (g.N - 1) // (q - 1))                      # generator of F_q^*
    Fq = [1]
    while len(Fq) < q - 1:
        Fq.append(g.mul(Fq[-1], z))
    seen, lines = set(), []
    for c in range(1, g.N):
        if c in seen:
            continue
        line = [g.mul(c, t) for t in Fq]
        seen.update(line)
        if all(mask[Linv(x)] for x in line):
            lines.append(symmetry.rref([Linv(x) for x in line]))
    return lines


def cmd_merge(args):
    meta, mask, cols, omin, reps = setup(args.dir)
    n = meta["n"]
    done = done_indices(args.dir)
    F = frontier(done, len(reps))
    k, c = args.ascending
    if k < F - 1:
        raise SystemExit(f"not complete: lanes cover indices >= {F}, ascending run reached {k}")
    G = [list(cols)]
    for _ in range(symmetry.group_order(cols, n) - 2):
        G.append([int(x) for x in symmetry.apply_linear(cols, np.array(G[-1]))])
    late = set()
    for i, S in done.items():
        if i > k:
            for B in S:
                B = tuple(B)
                late.add(B)
                for gm in G:
                    late.add(symmetry.rref(symmetry.apply_linear(gm, np.array(B))))
    total = c + len(late)
    lines = spread_lines_in_mask(meta, mask)
    res = {"n": n, "a": meta["a"], "nonbent": meta["nonbent"], "representatives": len(reps),
           "ascending_run": {"last_index": k, "subspaces": c}, "lanes_frontier": F,
           "subspaces_dim_m": total, "spread_lines_in_NB": len(lines),
           "all_spread_lines": total == len(lines),
           "bases_dim_m": [list(B) for B in lines] if total == len(lines) else None,
           "verdict": "none (necessary condition)" if total <= 1 else "needs Theorem 8"}
    print(json.dumps(res, indent=1))
    if args.out:
        from common import save
        save(args.out, {"rows": [res]})


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--a", type=int, required=True)
    s.add_argument("--dir", required=True)
    s = sub.add_parser("run")
    s.add_argument("--dir", required=True)
    s.add_argument("--lanes", required=True, help=f"e.g. 0-5 or 0,3,7 (of {NLANES})")
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.add_argument("--indices", default=None,
                   help="file with the indices to process, in this order (overrides the lane order)")
    s.add_argument("--tag", default="", help="suffix of the output files, e.g. _b for a helper run")
    s = sub.add_parser("status")
    s.add_argument("--dir", required=True)
    s = sub.add_parser("merge")
    s.add_argument("--dir", required=True)
    s.add_argument("--ascending", type=int, nargs=2, default=[-1, 0], metavar=("K", "C"))
    s.add_argument("--out", default=None)
    args = p.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "status": cmd_status, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
