"""
Check that the results in results/*.json support every number reported in the paper.

    python3 verify.py            # after running the experiments (see README.md)

Each check names the paper item it supports. Exit status 1 if any check fails or a
results file is missing.
"""
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
failures = []


def load(name):
    path = os.path.join(R, f"{name}.json")
    if not os.path.exists(path):
        failures.append(f"missing results/{name}.json")
        return None
    return json.load(open(path))


def check(item, cond, detail=""):
    print(f"[{'ok' if cond else 'FAIL'}] {item}{(' - ' + detail) if detail else ''}")
    if not cond:
        failures.append(item)


e1 = load("e01_inverse_function")
if e1:
    for r in e1["flats"]:
        check(f"Prop. 3 (n={r['n']}): cost 2^n - 1, bad 2-flats = GF(4)-lines",
              r["cost"] == r["expected_cost"] and r["bad_flats_are_gf4_lines"]
              and r["bad_flats"] == r["gf4_lines"])
    t = e1["transpositions_n8"]
    check("Sec. 3: no improving transposition of x^254 (n=8)", t["improving"] == 0)
    check("Sec. 3: 255 neutral transpositions, all involving 0", t["neutral"] == 255
          and t["neutral_all_involve_0"])
    check("Sec. 3: minimal positive cost change 222", t["min_positive_delta"] == 222)
    check("Sec. 3: no improvement two transpositions away", t["second_step_min_delta"] >= 0)
    check("Sec. 3: annealing never below 255",
          min(r["best_cost"] for r in e1["annealing_n8"]["runs"]) == 255)
    rnd = [r["best_cost"] for r in e1.get("annealing_random_n8", {}).get("runs", [])]
    check("Sec. 3: annealing from random permutations stalls far above 255 (about 5,000)",
          bool(rnd) and min(rnd) > 4000, str(sorted(rnd)[:3]))

e2 = load("e02_structured_constructions")
if e2:
    b = {r["k"]: r for r in e2["butterflies"]}
    check("Table 1: 6 APN butterflies for k = 3", b[3]["apn"] == 6)
    check("Table 1: best butterfly cost 8280 for n = 8", b[4]["min_cost"] == 8280)
    if 5 in b:
        check("Table 1: best butterfly cost 71424 for n = 10", b[5]["min_cost"] == 71424)
    check("Table 1: cyclotomic maps (l = 3, n = 8) never below 255",
          e2["cyclotomic"]["min_cost"] == 255)
    s = e2["switched_inverse"]
    check("Table 1: switched inverse: 33660 permutations, next cost after 255 is 4155",
          s["permutations"] == 33660 and s["min_cost_other_than_inverse_value"] == 4155)

e3 = load("e03_ccz_validation")
if e3:
    for r in e3["rows"]:
        if "dual_count" in r:
            check(f"Table 3: dual = exhaustive for {r['function']} (n={r['n']})",
                  r["dual_count"] == r["exhaustive_count"]
                  and r["dual_verdict"] == r["exhaustive_verdict"],
                  f"{r['exhaustive_count']} spaces, {r['exhaustive_verdict']}")
    kim = next(r for r in e3["rows"] if r["function"].startswith("Kim"))
    check("Table 3: Kim function gives an APN permutation",
          kim["dual_verdict"] == "permutation" and kim["dual_permutation_is_apn"])
    ep = next(r for r in e3["rows"] if r["function"].startswith("Edel-Pott"))
    check("Table 3: Edel-Pott function (degree 3) not CCZ to a permutation",
          ep["degree"] == 3 and ep["exhaustive_verdict"] == "none")

e4 = load("e04_n10_known_classes")
if e4:
    check("Table 4: 20 known quadratic APN classes for n = 10", e4["classes"] == 20,
          str(e4["classes_per_family"]))
    check("Table 4: none CCZ-equivalent to a permutation",
          all(r["verdict"] == "none" for r in e4["rows"]))
    small = sum(r["max_dim_NB"] < 5 for r in e4["rows"])
    check("Table 4: 13 classes excluded by the counting criterion (M < 5)", small == 13,
          f"{small}")
    inc = sorted(r["family"] for r in e4["rows"] if r.get("condition2_inconclusive"))
    check("Sec. 8.4: necessary condition inconclusive exactly for the 2 Gold classes and 1 Beierle-Leander class",
          inc == ["Beierle-Leander", "Gold", "Gold"], str(inc))
    ai = e4.get("all_instances", {})
    check("Table 4: every distinct APN instance tested, none CCZ to a permutation",
          ai.get("tested", 0) > 0 and set(ai.get("verdicts", {})) == {"none"}, str(ai))
    check("Sec. 8.4: Beierle-Leander #2 and #4 belong to F13",
          [r["in_family_classes"] for r in e4["beierle_leander"]] == [False, True, False, True, False])

e5 = load("e05_switching_walks")
if e5:
    exp = {"6": 13, "8": 17, "9": 6, "10": 20}
    for n, c in exp.items():
        r = e5[n]
        check(f"Table 5: n={n} walk closes with {c} quadratic classes",
              r["walk_complete"] and r["classes_reached"] == c, str(r["classes_reached"]))
    check("Table 5: non-quadratic switchings exist for n = 6 (Edel-Pott)", len(e5["6"]["nonquadratic"]) > 0)
    for n in ("8", "9", "10"):
        check(f"Table 5: no non-quadratic switching for n = {n}", not e5[n]["nonquadratic"])
    v6 = [r["verdict"] for r in e5["6"]["new_class_ccz"]]
    check("Sec. 5: n = 6 walk finds exactly one class CCZ-equivalent to a permutation (Kim)",
          v6.count("permutation") == 1, str(v6.count("permutation")))
    nq6 = {x["source"] for x in e5["6"]["nonquadratic"]}
    check("Sec. 5: n = 6, five classes with non-quadratic switchings, incl. the Edel-Pott source",
          len(nq6) == 5 and any("x^17" in s for s in nq6), str(len(nq6)))
    for n in ("8", "10"):
        check(f"Table 5: new classes for n = {n} not CCZ to permutations",
              all(r["verdict"] == "none" for r in e5[n]["new_class_ccz"]))

e6 = load("e06_quadratic_dfs")
if e6:
    rows = {r["n"]: r for r in e6["rows"]}
    check("Sec. 8.7: DFS reaches all 13 quadratic APN classes for n = 6", rows[6]["classes"] == 13)
    check("Sec. 8.7: DFS finds no solution for n = 8 within the budget", rows[8]["functions"] == 0)

e7 = load("e07_n10_sampling")
if e7:
    check("Sec. 8.7: symmetric sampling for n = 10 finds no class outside the 20 known",
          len(e7["new_classes"]) == 0, f"{e7['apn']} APN functions")

e8 = load("e08_tree_search")
if e8:
    check("Sec. 8.7: tree-search sweep, 69 of 195 classes finished",
          e8["sweep"]["classes"] == 195 and e8["sweep"]["finished"] == 69)
    check("Sec. 8.7: every tree-search solution tested, none CCZ to a permutation",
          set(e8.get("ccz_all_solutions", {})) == {"none"}, str(e8.get("ccz_all_solutions")))
    check("Sec. 8.7: deterministic + randomized tree search, 4638 APN functions in 16 classes",
          e8["apn"] == 4638 and e8["classes"] == 16, f"{e8['apn']}, {e8['classes']}")
    check("Sec. 8.7: all tree-search solutions in known classes", not e8["new_classes"],
          f"{e8['apn']} APN functions, {e8['classes']} classes")

e9 = load("e09_f4_family")
if e9:
    rows = {(r["n"], r["a"]): r for r in e9["rows"]}
    for n in (10, 14):
        for a in (1, 2, 4):
            r = rows[(n, a)]
            check(f"Table 2: n={n}, a={a}: exactly one m-dim subspace, a spread line, = aF_q iff a cube",
                  r["subspaces_dim_m"] == 1 and r["subspaces_dim_m_plus_1"] == 0 and r["all_spread_lines"]
                  and r["contains_a_line"] == r["a_is_cube"] and r["verdict"] == "none")
    r = rows[(6, 1)]
    check("Table 2: n=6, a=1: three spread lines incl. aF_q, condition inconclusive, decided: none",
          r["subspaces_dim_m"] == 3 and r["all_spread_lines"] and r["contains_a_line"]
          and r["condition_inconclusive"] and r["verdict"] == "none")
    check("Table 2: n=6, a not a cube: no m-dim subspace",
          rows[(6, 2)]["subspaces_dim_m"] == 0 and rows[(6, 4)]["subspaces_dim_m"] == 0)
    cs = {c["n"]: c for c in e9["cube_subspaces"]}
    check("Sec. 7: m = 5, 7: all m- and (m-1)-dim subspaces of the cubes lie in spread lines",
          cs[10]["dim_5"] == cs[10]["dim_5_inside_a_line"] == 11 and cs[10]["dim_4"] == cs[10]["dim_4_inside_a_line"] == 341
          and cs[14]["dim_7"] == cs[14]["dim_7_inside_a_line"] == 43 and cs[14]["dim_6"] == cs[14]["dim_6_inside_a_line"] == 5461)
    check("Sec. 7: m = 3: only 21 of 28 two-dim subspaces of the cubes lie in spread lines",
          cs[6]["dim_2"] == 28 and cs[6]["dim_2_inside_a_line"] == 21)

def e10_rows(res):
    """Rows of E10 with the dimension keys of older runs (subspaces_dim_9/10) renamed."""
    for r in res["rows"]:
        if "subspaces_dim_9" in r:
            r["subspaces_dim_m"], r["subspaces_dim_m_plus_1"] = r["subspaces_dim_9"], r["subspaces_dim_10"]
    return res["rows"]


e10 = load("e10_f4_n18")
if e10:
    rows = {r["a"]: r for r in e10_rows(e10)}
    r = rows[1]
    check("Table 2 / Cor. 11: n=18, a=1: 3 subspaces, 3 spanning pairs, decided by Theorem 8: none",
          r["subspaces_dim_m"] == 3 and r["subspaces_dim_m_plus_1"] == 0 and r["spanning_pairs"] == 3
          and r["verdict"].startswith("none (Theorem 8)"))
    for a in (2, 4):
        check(f"Table 2: n=18, a={a}: no 9-dim subspace, excluded by the necessary condition",
              rows[a]["subspaces_dim_m"] == 0 and rows[a]["subspaces_dim_m_plus_1"] == 0
              and rows[a]["verdict"].startswith("none"))

e10b = load("e10_f4_n18_a1")
if e10b:
    r = e10b["rows"][0]
    check("Table 2: n=18, a=1: the three subspaces are spread lines, one of them aF_q",
          r["all_spread_lines"] and r["contains_a_line"])
    check("Table 2: n=18, a=1 (independent local run): decided by Theorem 8, none",
          r["verdict"].startswith("none (Theorem 8)"))

e22 = load("e10_f4_n22_a1_m")
if e22:
    r = e22["rows"][0]
    check("Sec. 7 (E10, n = 22): a = 1 has exactly one 11-dim subspace, the spread line aF_q",
          r["n"] == 22 and r["subspaces_dim_m"] == 1 and r["all_spread_lines"] and r["contains_a_line"]
          and r["verdict"].startswith("none"))

e11 = load("e11_n10_recent_instances")
if e11:
    sm, cps = e11["summary"], e11["summary"]["classes_per_source"]
    check("Sec. 8.4 (E11): every APN instance is certified EA-equivalent to the reference of its class",
          sm["certified"] == sm["apn_instances"] == 373, f"{sm['certified']}/{sm['apn_instances']}")
    bik = [r for r in e11["rows"] if r["source"] == "Budaghyan-Ivkovic-Kaleyski"]
    check("Sec. 8.4 (E11): the six functions of Budaghyan-Ivkovic-Kaleyski are APN with the printed spectra",
          len(bik) == 12 and all(r["apn"] and r["spectrum_as_printed"] for r in bik))
    check("Sec. 8.4 (E11): Table 2 #2 of Budaghyan-Ivkovic-Kaleyski is EA-equivalent to x^3 + beta x^36",
          cps["Budaghyan-Ivkovic-Kaleyski: Table 2 #2"] == ["F13 i=3 k=2 a=236 b=0 c=0"])
    check("Sec. 8.4 (E11): the instances of Zheng et al. lie in E4 classes",
          all(not c.startswith("NEW") for k, v in cps.items() if k.startswith("Zheng") for c in v))
    check("Sec. 8.4 (E11): Goloğlu F1 / CLV family 1 and Goloğlu F2 give two F13-type classes each",
          cps["Gologlu: Gologlu F1"] == cps["Calderini-Li-Villa: CLV family 1"]
          == ["F13 i=3 k=0 a=1 b=236 c=237", "F13 i=3 k=0 a=236 b=236 c=237"]
          and cps["Gologlu: Gologlu F2"] == ["F13 i=3 k=0 a=1 b=1 c=236", "F13 i=3 k=0 a=1 b=236 c=1"])
    check("Sec. 8.4 (E11): Carlet's family gives the two F3 classes",
          cps["Carlet: Carlet family"] == ["F3 i=1 c=1014 s=807", "F3 i=2 c=941 s=1007"])
    b1 = "NEW: Budaghyan-Ivkovic-Kaleyski Table 2 #1 beta = 236"
    check("Sec. 8.4 (E11): Li-Zhou-Li-Qu and CLV family 2 are in the class of BIK Table 2 #1",
          cps["Li-Zhou-Li-Qu: Li-Zhou-Li-Qu Thm 5"] == cps["Calderini-Li-Villa: CLV family 2"] == [b1])
    check("Table 4 (E11): five further classes, none CCZ-equivalent to a permutation, all with M = 4",
          sm["new_classes"] == 5 and all(t["verdict"] == "none" and t["max_dim_NB"] == 4
                                         for t in e11["new_classes"]))
    check("Sec. 8.4 (E11): Beierle-Leander #2 and #4 are EA-equivalent to F13-type functions",
          len(e11["e4_invariant_matches"]) == 2 and all(p["ea_equivalent"] for p in e11["e4_invariant_matches"]))
    sw = e11["switching"]
    check("Table 5 (E11): switching from the five further classes is complete and reaches no other class",
          sw["complete"] and sw["classes_reached"] == sw["in_new_classes"] == 5 and not sw["nonquadratic"], str(sw))
    check("Sec. 2 (E11): the representative of Yu et al. is EA-equivalent to x^3 + Tr(x^9)",
          cps["Yu-Li-Ichanska-Kaleyski: x^513+x^192+x^96+x^48+x^33+x^18+x^12"] == ["F4 a=1"])

print()
if failures:
    print(f"{len(failures)} check(s) failed or missing:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
