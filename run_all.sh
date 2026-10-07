#!/bin/sh
# Re-run every experiment of the paper and check the reported numbers.
#   ./run_all.sh          full budgets (several hours on a multi-core machine; E8 separately)
#   ./run_all.sh --quick  reduced budgets (about 30 minutes)
set -e
cd "$(dirname "$0")"
python3 -m pytest tests -q
cd experiments
for e in e01_inverse_function e02_structured_constructions e03_ccz_validation \
         e04_n10_known_classes e05_switching_walks e06_quadratic_dfs e07_n10_sampling e09_f4_family \
         e11_n10_recent_instances; do
    echo "== $e"
    python3 $e.py "$@"
done
echo "== e08 (tree search) and e10 (n = 18, many cores) are run separately, see README.md"
cd ..
python3 verify.py
