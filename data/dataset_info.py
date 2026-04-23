"""
dataset_info.py
---------------
Reads dataset.json and prints a structured summary of its contents.
Does NOT perform any eigenvalue calculations or matrix operations.
Pure metadata and counting only.
"""

import json
import os
from collections import Counter, defaultdict

# ── Load ──────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")

if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] File not found: {DATASET_PATH}")
    exit(1)

with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

total = len(dataset)


# ── Helpers ───────────────────────────────────────────────────────────────────


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def subsection(title):
    print(f"\n  {title}")
    print(f"  {'-'*40}")


def bar(label, count, total, width=20):
    filled = int(width * count / total) if total else 0
    pct = 100 * count / total if total else 0
    return f"  {label:<20}  {count:>3}  {'█'*filled:<{width}}  {pct:>5.1f}%"


# ── 1. Overview ───────────────────────────────────────────────────────────────

section("OVERVIEW")
print(f"\n  Total problems   : {total}")
print(f"  File             : {os.path.abspath(DATASET_PATH)}")
print(f"  File size        : {os.path.getsize(DATASET_PATH):,} bytes")


# ── 2. IDs ────────────────────────────────────────────────────────────────────

section("RECORD IDs")
ids = [r["id"] for r in dataset]
print(f"\n  First 5  : {ids[:5]}")
print(f"  Last  5  : {ids[-5:]}")
duplicates = [id_ for id_, cnt in Counter(ids).items() if cnt > 1]
print(f"  Duplicate IDs : {duplicates if duplicates else 'None'}")


# ── 3. Dimension breakdown ────────────────────────────────────────────────────

section("BY MATRIX DIMENSION")
dim_counts = Counter(r["equation"]["dimension"] for r in dataset)
subsection("Count per dimension")
for dim in sorted(dim_counts):
    print(bar(f"{dim}×{dim}", dim_counts[dim], total))


# ── 4. Category breakdown ─────────────────────────────────────────────────────

section("BY CATEGORY")
cat_counts = Counter(r["metadata"]["category"] for r in dataset)
subsection("Count per category")
for cat in sorted(cat_counts):
    print(bar(cat, cat_counts[cat], total))


# ── 5. Difficulty breakdown ───────────────────────────────────────────────────

section("BY DIFFICULTY")
diff_counts = Counter(r["metadata"]["difficulty"] for r in dataset)
subsection("Count per difficulty level")
for diff in ["EASY", "MEDIUM", "HARD"]:
    print(bar(diff, diff_counts.get(diff, 0), total))


# ── 6. Decimal type breakdown ─────────────────────────────────────────────────

section("BY DECIMAL TYPE")
dec_counts = Counter(r["metadata"]["decimal_type"] for r in dataset)
subsection("Count per decimal precision")
for dt in sorted(dec_counts):
    label = {"1": "1 decimal place", "2": "2 decimal places", "m": "mixed"}.get(dt, dt)
    print(bar(label, dec_counts[dt], total))


# ── 7. Boolean flags ──────────────────────────────────────────────────────────

section("BOOLEAN FLAG SUMMARY")
flags = ["is_singular", "has_repeated", "is_symmetric", "is_nilpotent"]
subsection("How many problems have each flag set to True")
for flag in flags:
    true_count = sum(1 for r in dataset if r["metadata"][flag])
    false_count = total - true_count
    print(f"  {flag:<20}  True={true_count:>3}  False={false_count:>3}")


# ── 8. Rank and nullity ───────────────────────────────────────────────────────

section("RANK AND NULLITY")
rank_counts = Counter(r["metadata"]["rank"] for r in dataset)
nullity_counts = Counter(r["metadata"]["nullity"] for r in dataset)

subsection("Rank distribution")
for rank in sorted(rank_counts):
    print(bar(f"rank = {rank}", rank_counts[rank], total))

subsection("Nullity distribution")
for nullity in sorted(nullity_counts):
    print(bar(f"nullity = {nullity}", nullity_counts[nullity], total))


# ── 9. Eigenvalue counts ──────────────────────────────────────────────────────

section("EIGENVALUE COUNT PER PROBLEM")
ev_counts = Counter(len(r["result"]["eigenvalues"]) for r in dataset)
subsection("How many eigenvalues per problem (should equal dimension)")
for n in sorted(ev_counts):
    print(bar(f"{n} eigenvalues", ev_counts[n], total))


# ── 10. Eigenvector counts ────────────────────────────────────────────────────

section("EIGENVECTOR COUNT PER PROBLEM")
evec_counts = Counter(len(r["result"]["eigenvectors"]) for r in dataset)
subsection("How many eigenvectors per problem")
for n in sorted(evec_counts):
    print(bar(f"{n} eigenvectors", evec_counts[n], total))


# ── 11. Intermediate steps ────────────────────────────────────────────────────

section("INTERMEDIATE STEPS")
step_counts = Counter(len(r["intermediate_steps"]) for r in dataset)
all_step_names = []
for r in dataset:
    all_step_names.extend(s["step"] for s in r["intermediate_steps"])
step_type_counts = Counter(all_step_names)

subsection("Number of steps per problem")
for n in sorted(step_counts):
    print(bar(f"{n} steps", step_counts[n], total))

subsection("Total occurrences of each step type across all problems")
for step_name in sorted(step_type_counts):
    print(f"  {step_name:<45}  {step_type_counts[step_name]:>4}")


# ── 12. Cross-tab: dimension × category ──────────────────────────────────────

section("CROSS-TAB: DIMENSION × CATEGORY")
cross = defaultdict(int)
for r in dataset:
    cross[(r["equation"]["dimension"], r["metadata"]["category"])] += 1

all_dims = sorted(set(r["equation"]["dimension"] for r in dataset))
all_cats = sorted(set(r["metadata"]["category"] for r in dataset))

print(f"\n  {'category':<18}", end="")
for dim in all_dims:
    print(f"  {dim}×{dim}", end="")
print()
print(f"  {'-'*50}")
for cat in all_cats:
    print(f"  {cat:<18}", end="")
    for dim in all_dims:
        print(f"  {cross[(dim, cat)]:>4}", end="")
    print()


# ── 13. Cross-tab: dimension × difficulty ─────────────────────────────────────

section("CROSS-TAB: DIMENSION × DIFFICULTY")
cross2 = defaultdict(int)
for r in dataset:
    cross2[(r["equation"]["dimension"], r["metadata"]["difficulty"])] += 1

diffs = ["EASY", "MEDIUM", "HARD"]
print(f"\n  {'difficulty':<18}", end="")
for dim in all_dims:
    print(f"  {dim}×{dim}", end="")
print()
print(f"  {'-'*50}")
for diff in diffs:
    print(f"  {diff:<18}", end="")
    for dim in all_dims:
        print(f"  {cross2[(dim, diff)]:>4}", end="")
    print()


# ── 14. Sample record preview ─────────────────────────────────────────────────

section("SAMPLE RECORD PREVIEW (first record, no values shown)")
r = dataset[0]
print(f"\n  id          : {r['id']}")
print(f"  dimension   : {r['equation']['dimension']}")
print(f"  category    : {r['metadata']['category']}")
print(f"  difficulty  : {r['metadata']['difficulty']}")
print(f"  decimal_type: {r['metadata']['decimal_type']}")
print(f"  rank        : {r['metadata']['rank']}")
print(f"  nullity     : {r['metadata']['nullity']}")
print(f"  is_singular : {r['metadata']['is_singular']}")
print(f"  has_repeated: {r['metadata']['has_repeated']}")
print(f"  is_symmetric: {r['metadata']['is_symmetric']}")
print(f"  is_nilpotent: {r['metadata']['is_nilpotent']}")
print(f"  num eigenvalues  : {len(r['result']['eigenvalues'])}")
print(f"  num eigenvectors : {len(r['result']['eigenvectors'])}")
print(f"  num steps        : {len(r['intermediate_steps'])}")
print(f"  step names       : {[s['step'] for s in r['intermediate_steps']]}")

print(f"\n{'='*55}")
print(f"  END OF REPORT  |  Total problems: {total}")
print(f"{'='*55}\n")
