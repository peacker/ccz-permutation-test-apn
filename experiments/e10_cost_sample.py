"""
E10, cost of the n = 22 searches: CPU time of the orbit-representative searches for two values
of a, on a matched sample.

For each a, the orbit representatives r_0 < r_1 < ... are those of e10_lanes.py (directories
prepared with `e10_lanes.py prepare`). The sample takes, for k = 0..K-1, the relative position
p_k = (k + u) / K with one fixed random u, and the representative at index floor(p_k * R) of each
list (R = number of representatives). Since the cost of a search depends strongly on its position
in the list, this pairs searches of comparable position. The CPU time of every search is measured
in the worker. The output gives the ratio of the per-representative costs and, for each a, the
estimated total CPU time R * (mean sample time).

    python3 e10_cost_sample.py --dirs D1 D2 --k 800 --workers 200 --out e10_cost_sample_n22
"""
import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import common  # noqa: F401  (puts the artifact root on sys.path)
from common import save
from apnsearch import symmetry
from e10_lanes import setup

STATE = {}


def _init(dirs):
    for d in dirs:
        meta, mask, cols, omin, reps = setup(d)
        STATE[d] = (meta, mask, omin, reps)


def timed(args):
    d, i = args
    meta, mask, omin, reps = STATE[d]
    symmetry._init(mask, omin)
    t = time.process_time()
    found = symmetry._search_from((int(reps[i]), meta["m"], meta["n"]))
    return d, i, time.process_time() - t, len(found)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs=2, required=True)
    p.add_argument("--k", type=int, default=800)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    u = random.Random(args.seed).random()
    info = {}
    for d in args.dirs:
        meta, mask, cols, omin, reps = setup(d)
        info[d] = {"a": meta["a"], "representatives": len(reps)}
    jobs = []
    for k in range(args.k):
        pos = (k + u) / args.k
        for d in args.dirs:
            jobs.append((pos, d, int(pos * info[d]["representatives"])))
    jobs.sort()                                   # expensive (early) positions first
    times = {d: {} for d in args.dirs}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.dirs,)) as ex:
        futs = [ex.submit(timed, (d, i)) for _, d, i in jobs]
        for n_done, fut in enumerate(as_completed(futs), 1):
            d, i, cpu, nfound = fut.result()
            times[d][i] = cpu
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(jobs)} [{time.time() - t0:.0f}s]", flush=True)
    rows = []
    for d in args.dirs:
        ts = list(times[d].values())
        mean = sum(ts) / len(ts)
        R = info[d]["representatives"]
        rows.append({"a": info[d]["a"], "representatives": R, "sample": len(ts),
                     "mean_cpu_seconds": round(mean, 3), "max_cpu_seconds": round(max(ts), 1),
                     "estimated_total_core_hours": round(R * mean / 3600, 2), "_total": R * mean})
        print({k: v for k, v in rows[-1].items() if k != '_total'}, flush=True)
    res = {"k": args.k, "u": u, "rows": rows,
           "ratio_per_representative": round(rows[1]["mean_cpu_seconds"] / rows[0]["mean_cpu_seconds"], 3),
           "ratio_total": round(rows[1].pop("_total") / rows[0].pop("_total"), 3),
           "wall_seconds": round(time.time() - t0)}
    print(res, flush=True)
    if args.out:
        save(args.out, res)


if __name__ == "__main__":
    main()
