"""Shared helpers for the experiment scripts."""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")


def parser(doc):
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    p.add_argument("--quick", action="store_true", help="reduced budget (minutes instead of hours)")
    return p


def save(name, data):
    os.makedirs(RESULTS, exist_ok=True)
    data = {"experiment": name, "date": time.strftime("%Y-%m-%d %H:%M:%S"), **data}
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=1, default=int)
    print(f"-> results/{name}.json")
