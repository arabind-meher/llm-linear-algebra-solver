import io
import json
import os
import random
import sys

import numpy as np

# Silence all prints during import and use
sys.stdout = io.StringIO()
from equation.eigen_problem import EigenProblem

sys.stdout = sys.__stdout__

random.seed(42)
np.random.seed(42)


def r(x, dp):
    return round(float(x), dp)


def silent_call(fn, *args, **kwargs):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return result


def make_problem(matrix, dim):
    p = EigenProblem(matrix=matrix, dimension=dim)
    silent_call(p.analyze_matrix)
    p.solve()
    A = np.array(matrix, dtype=float)
    for lam, vec in zip(p.eigenvalues, p.eigenvectors):
        v = np.array(vec, dtype=float)
        if np.linalg.norm(A @ v - lam * v) >= 1e-3:
            return None, None
    analysis = silent_call(p.analyze_matrix)
    return p, analysis


def analyze(p):
    old = sys.stdout
    sys.stdout = io.StringIO()
    result = p.analyze_matrix()
    sys.stdout = old
    return result


def rand_mat(n, dp):
    return [[r(random.uniform(-9.9, 9.9), dp) for _ in range(n)] for _ in range(n)]


def gen_distinct(n, dp):
    return rand_mat(n, dp)


def gen_repeated(n, dp):
    val = r(random.uniform(-5, 5), 1)
    others = []
    while len(others) < n - 2:
        v = r(random.uniform(-5, 5), 1)
        if v != val and v not in others:
            others.append(v)
    vals = [val, val] + others
    random.shuffle(vals)
    A = np.diag(vals).astype(float)
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i][j] = r(random.uniform(-0.1, 0.1), dp)
    return [[r(A[i][j], dp) for j in range(n)] for i in range(n)]


def gen_singular(n, dp):
    m = rand_mat(n, dp)
    m[-1] = m[0][:]  # last row = first row → det=0
    return m


def gen_sym_pd(n, dp):
    B = np.random.uniform(0.5, 2.0, (n, n))
    A = B.T @ B + n * np.eye(n)
    return [[r(A[i][j], dp) for j in range(n)] for i in range(n)]


def gen_nilpotent(n, dp):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            A[i][j] = r(random.uniform(0.5, 3.0), dp)
    return A


CHECKS = {
    "distinct": lambda a: not a["has_repeated"] and not a["is_singular"],
    "repeated": lambda a: a["has_repeated"],
    "singular": lambda a: a["is_singular"],
    "symmetric_pd": lambda a: a["is_symmetric"] and not a["is_singular"],
    "nilpotent": lambda a: a["is_nilpotent"],
}
GENS = {
    "distinct": gen_distinct,
    "repeated": gen_repeated,
    "singular": gen_singular,
    "symmetric_pd": gen_sym_pd,
    "nilpotent": gen_nilpotent,
}

# 50-matrix plan
PLAN = [
    (3, 6, "distinct", 1),
    (3, 5, "distinct", 2),
    (3, 4, "distinct", "m"),
    (3, 4, "repeated", 1),
    (3, 3, "repeated", 2),
    (3, 3, "singular", 1),
    (3, 2, "singular", "m"),
    (3, 2, "symmetric_pd", 1),
    (3, 1, "nilpotent", 1),
    (4, 4, "distinct", 1),
    (4, 4, "distinct", 2),
    (4, 3, "repeated", 1),
    (4, 2, "singular", 1),
    (4, 1, "singular", 2),
    (4, 2, "symmetric_pd", 2),
    (4, 1, "nilpotent", "m"),
    (5, 2, "distinct", 1),
    (5, 2, "distinct", 2),
    (5, 2, "repeated", 1),
    (5, 2, "singular", 1),
    (5, 1, "symmetric_pd", "m"),
    (5, 1, "nilpotent", 2),
]


def to_record(p, analysis, pid, cat, dp_label):
    steps = [
        {"step": "characteristic_polynomial_coefficients", "value": p.coefficients},
        {"step": "eigenvalues", "value": p.eigenvalues},
    ]
    for i, (sm, rr) in enumerate(zip(p.shifted_matrices, p.rref_matrices)):
        steps.append({"step": f"shifted_matrix_lambda_{i+1}", "value": sm})
        steps.append({"step": f"rref_lambda_{i+1}", "value": rr})
    return {
        "id": pid,
        "equation": {"matrix": p.matrix, "dimension": p.dimension},
        "metadata": {
            "category": cat,
            "decimal_type": str(dp_label),
            "difficulty": analysis["difficulty"],
            "rank": analysis["rank"],
            "nullity": analysis["nullity"],
            "is_singular": analysis["is_singular"],
            "has_repeated": analysis["has_repeated"],
            "is_symmetric": analysis["is_symmetric"],
            "is_nilpotent": analysis["is_nilpotent"],
        },
        "intermediate_steps": steps,
        "result": {"eigenvalues": p.eigenvalues, "eigenvectors": p.eigenvectors},
    }


dataset, counters = [], {}
print("Generating 50-matrix dataset...\n")

for dim, count, cat, dp_label in PLAN:
    gen = GENS[cat]
    check = CHECKS[cat]
    got, tries = 0, 0
    print(f"  {dim}×{dim}  {cat:<15}  target={count}", end="  ", flush=True)

    while got < count and tries < count * 300:
        tries += 1
        dp = random.choice([1, 2]) if dp_label == "m" else dp_label
        try:
            matrix = gen(dim, dp)
            p = EigenProblem(matrix=matrix, dimension=dim)
            analysis = analyze(p)
            if not check(analysis):
                continue
            p.solve()
            A = np.array(matrix, dtype=float)
            ok = all(
                np.linalg.norm(A @ np.array(v) - lam * np.array(v)) < 1e-3
                for lam, v in zip(p.eigenvalues, p.eigenvectors)
            )
            if not ok:
                continue
            key = f"{dim}_{cat}"
            counters[key] = counters.get(key, 0) + 1
            pid = f"{dim}x{dim}_{cat}_{counters[key]:03d}"
            dataset.append(to_record(p, analysis, pid, cat, dp_label))
            got += 1
        except Exception:
            continue

    print(f"got={got}")

os.makedirs("data", exist_ok=True)
with open(os.path.join("data", "dataset_v2.json"), "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\nTotal: {len(dataset)}")
dims = {}
cats = {}
diffs = {}
for rec in dataset:
    d = rec["equation"]["dimension"]
    c = rec["metadata"]["category"]
    diff = rec["metadata"]["difficulty"]
    dims[d] = dims.get(d, 0) + 1
    cats[c] = cats.get(c, 0) + 1
    diffs[diff] = diffs.get(diff, 0) + 1
for k, v in sorted(dims.items()):
    print(f"  {k}×{k}: {v}")
print()
for k, v in sorted(cats.items()):
    print(f"  {k:<15}: {v}")
print()
for k, v in sorted(diffs.items()):
    print(f"  {k}: {v}")
for k, v in sorted(dims.items()):
    print(f"  {k}×{k}: {v}")
print()
for k, v in sorted(cats.items()):
    print(f"  {k:<15}: {v}")
print()
for k, v in sorted(diffs.items()):
    print(f"  {k}: {v}")
