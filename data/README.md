# Dataset — Eigenvalue & Eigenvector Computation

55 decimal-valued square matrices (3×3, 4×4, 5×5) with fully verified ground-truth intermediate steps and final results.

---

## Files
 
```
data/
├── dataset.json        # 55 problems with full step-by-step ground truth
├── dataset_info.py     # standalone script: prints statistics, no calculations
└── README.md
 
scripts/
└── gen50.py            # dataset generation script
 
src/equation/
├── __init__.py         # exports EigenProblem
└── eigen_problem.py    # core class: constructs, solves, and verifies problems
```
 
---

## Problem Scope

- **Matrix sizes:** 3×3, 4×4, and 5×5 only
- **Entry type:** decimal values (1 decimal place, 2 decimal places, or mixed)
- **Eigenvalue type:** real only — no complex values
- **Scale:** 55 problems

---

## Dataset Statistics

### By Matrix Dimension

| Dimension | Count | Share |
|-----------|------:|------:|
| 3×3       |    28 | 50.9% |
| 4×4       |    17 | 30.9% |
| 5×5       |    10 | 18.2% |

### By Category

| Category     | Count | Share | Description |
|--------------|------:|------:|-------------|
| distinct     |    27 | 49.1% | All eigenvalues are distinct |
| repeated     |    10 | 18.2% | At least one eigenvalue appears more than once |
| singular     |    10 | 18.2% | Matrix is rank-deficient (at least one zero eigenvalue) |
| symmetric_pd |     5 |  9.1% | Symmetric positive definite (all positive eigenvalues) |
| nilpotent    |     3 |  5.5% | Strictly upper triangular (all eigenvalues zero) |

### By Difficulty

Difficulty is assigned based on structural properties: repeated eigenvalues (+2), singular (+2), zero eigenvalue (+1), non-symmetric (+1), nilpotent (+1). Score ≤1 = EASY, 2–3 = MEDIUM, ≥4 = HARD.

| Level  | Count | Share |
|--------|------:|------:|
| EASY   |    32 | 58.2% |
| MEDIUM |    10 | 18.2% |
| HARD   |    13 | 23.6% |

### By Decimal Type

| Type             | Count | Share |
|------------------|------:|------:|
| 1 decimal place  |    31 | 56.4% |
| 2 decimal places |    16 | 29.1% |
| mixed            |     8 | 14.5% |

### Boolean Flags

| Flag         | True | False |
|--------------|-----:|------:|
| is_singular  |   13 |    42 |
| has_repeated |   13 |    42 |
| is_symmetric |    5 |    50 |
| is_nilpotent |    3 |    52 |

### Dimension × Category Cross-Tab

| Category     | 3×3 | 4×4 | 5×5 |
|--------------|----:|----:|----:|
| distinct     |  15 |   8 |   4 |
| nilpotent    |   1 |   1 |   1 |
| repeated     |   5 |   3 |   2 |
| singular     |   5 |   3 |   2 |
| symmetric_pd |   2 |   2 |   1 |

### Dimension × Difficulty Cross-Tab

| Difficulty | 3×3 | 4×4 | 5×5 |
|------------|----:|----:|----:|
| EASY       |  17 |  10 |   5 |
| MEDIUM     |   5 |   3 |   2 |
| HARD       |   6 |   4 |   3 |

---

## Record Format

Each record in `dataset.json` follows this structure:

```json
{
  "id": "3x3_distinct_001",
  "equation": {
    "matrix": [[1.0, 4.0, 7.0], [2.0, 3.0, 3.0], [4.0, 9.0, 0.0]],
    "dimension": 3
  },
  "metadata": {
    "category": "distinct",
    "decimal_type": "1",
    "difficulty": "EASY",
    "rank": 3,
    "nullity": 0,
    "is_singular": false,
    "has_repeated": false,
    "is_symmetric": false,
    "is_nilpotent": false
  },
  "intermediate_steps": [
    {"step": "characteristic_polynomial_coefficients", "value": [...]},
    {"step": "eigenvalues",                            "value": [...]},
    {"step": "shifted_matrix_lambda_1",                "value": [...]},
    {"step": "rref_lambda_1",                          "value": [...]},
    ...
  ],
  "result": {
    "eigenvalues":  [...],
    "eigenvectors": [...]
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier: `{dim}x{dim}_{category}_{seq}` |
| `equation.matrix` | list of lists | The input matrix (float entries) |
| `equation.dimension` | int | Matrix size n (3, 4, or 5) |
| `metadata.category` | string | One of: distinct, repeated, singular, symmetric_pd, nilpotent |
| `metadata.decimal_type` | string | `"1"`, `"2"`, or `"m"` (mixed) |
| `metadata.difficulty` | string | EASY, MEDIUM, or HARD |
| `metadata.rank` | int | Matrix rank |
| `metadata.nullity` | int | Dimension of null space (= n − rank) |
| `metadata.is_singular` | bool | True if rank < n |
| `metadata.has_repeated` | bool | True if any eigenvalue appears more than once (rounded to 4 dp) |
| `metadata.is_symmetric` | bool | True if A = Aᵀ |
| `metadata.is_nilpotent` | bool | True if Aⁿ ≈ 0 |
| `intermediate_steps` | list | Ordered sequence of named computation steps (see below) |
| `result.eigenvalues` | list of float | Final eigenvalues, sorted descending |
| `result.eigenvectors` | list of list | One eigenvector per eigenvalue, normalized |

### Intermediate Steps

Each problem includes a fixed sequence of named steps. The number of steps scales with dimension.

| Step name | Occurs | Description |
|-----------|--------|-------------|
| `characteristic_polynomial_coefficients` | all 55 | Coefficients of det(A − λI), descending order |
| `eigenvalues` | all 55 | Real eigenvalues sorted descending |
| `shifted_matrix_lambda_k` | all 55 | (A − λₖI) for each eigenvalue k |
| `rref_lambda_k` | all 55 | Row-reduced echelon form of (A − λₖI) |

Steps per problem: **8** for 3×3, **10** for 4×4, **12** for 5×5.

---

## ID Scheme

IDs follow the pattern `{n}x{n}_{category}_{seq}` with a 3-digit zero-padded sequence number, e.g.:

- `3x3_distinct_001` through `3x3_distinct_015`
- `4x4_singular_001` through `4x4_singular_003`
- `5x5_nilpotent_001`

No duplicate IDs exist in the dataset.

---

## Generation

All problems were generated programmatically (`gen50.py` + `eigen_problem.py`) with fixed random seeds (`seed=42`) for reproducibility. Each category has a dedicated construction strategy that guarantees the intended structural property before any computation runs. Matrices that produce complex eigenvalues are rejected immediately at construction time.

After construction, every matrix goes through a two-stage acceptance check:

1. **Category check** — the matrix is analyzed to confirm it actually has the expected property (e.g. `has_repeated`, `is_singular`). Rounding decimal entries can occasionally destroy the property, so this step discards those cases.
2. **Residual check** — each computed eigenvector `v` must satisfy `‖Av − λv‖ < 1e-3`. Any matrix where a pair fails this threshold is discarded.

`Only matrices that pass both checks are kept.`

### Category construction

- **distinct** — random decimal matrix with entries in `[−9.9, 9.9]`; accepted only if all eigenvalues are distinct and the matrix is non-singular
- **repeated** — starts from a diagonal matrix with one eigenvalue intentionally duplicated, then adds small off-diagonal noise (±0.1) to make it non-trivial
- **singular** — random matrix with the last row overwritten to be a copy of the first row, which guarantees `det = 0` and forces at least one zero eigenvalue
- **symmetric_pd** — constructed as `BᵀB + nI` where `B` is a matrix; `BᵀB` is always symmetric and adding `nI` ensures all eigenvalues are strictly positive
- **nilpotent** — strictly upper triangular matrix (only positions above the diagonal are filled); all eigenvalues are zero by definition

### Regenerating the dataset
 
The dataset is already included as `data/dataset.json` and does not need to be regenerated. If you do want to regenerate it:
 
```bash
uv run python scripts/gen50.py
```
 
This overwrites `data/dataset.json` in place. The script must be run from the project root. Output is written relative to the script's working directory, so running it from anywhere else will place `dataset.json` in the wrong location.
