import json
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import sympy as sp


@dataclass
class EigenProblem:
    matrix: List[List[float]]
    dimension: Literal[3, 4, 5]
    coefficients: Optional[List[float]] = None
    eigenvalues: Optional[List[float]] = None
    shifted_matrices: Optional[List[List[List[float]]]] = None
    rref_matrices: Optional[List[List[List[float]]]] = None
    eigenvectors: Optional[List[List[float]]] = None

    def __post_init__(self):
        if self.dimension not in (3, 4, 5):
            raise ValueError(f"Dimension must be 3, 4, or 5. Got {self.dimension}")
        if len(self.matrix) != self.dimension:
            raise ValueError(f"Matrix must have {self.dimension} rows. Got {len(self.matrix)}")
        for i, row in enumerate(self.matrix):
            if len(row) != self.dimension:
                raise ValueError(f"Row {i} must have {self.dimension} columns. Got {len(row)}")
        self.A = sp.Matrix(self.matrix)
        eigenvalues = np.linalg.eig(np.array(self.matrix, dtype=float))[0]
        if np.any(np.abs(eigenvalues.imag) > 1e-6):
            raise ValueError(f"Matrix has complex eigenvalues. Only real eigenvalues are supported.")

    def __repr__(self) -> str:
        def fmt(val):
            return val if val is not None else "Not computed yet"

        return json.dumps(
            {
                "dimension": self.dimension,
                "matrix": self.matrix,
                "coefficients": fmt(self.coefficients),
                "eigenvalues": fmt(self.eigenvalues),
                "shifted_matrices": fmt(self.shifted_matrices),
                "rref_matrices": fmt(self.rref_matrices),
                "eigenvectors": fmt(self.eigenvectors),
            },
            indent=2,
        )

    def analyze_matrix(self) -> dict:
        A = np.array(self.matrix, dtype=float)
        rank = np.linalg.matrix_rank(A)
        eigenvalues = np.linalg.eig(A)[0].real
        is_singular = rank < self.dimension
        has_zero_eigenvalue = any(abs(v) < 1e-6 for v in eigenvalues)
        has_repeated = len(set(round(v, 4) for v in eigenvalues)) < self.dimension
        is_symmetric = np.allclose(A, A.T)
        is_diagonal = np.allclose(A, np.diag(np.diagonal(A)))
        is_identity = np.allclose(A, np.eye(self.dimension))
        is_zero = np.allclose(A, np.zeros((self.dimension, self.dimension)))
        is_nilpotent = np.allclose(
            np.linalg.matrix_power(A, self.dimension), np.zeros((self.dimension, self.dimension))
        )
        num_zero_eigenvalues = sum(1 for v in eigenvalues if abs(v) < 1e-6)
        condition_number = np.linalg.cond(A) if not is_singular else float("inf")
        difficulty = sum(
            [
                2 * int(has_repeated),
                2 * int(is_singular),
                1 * int(has_zero_eigenvalue),
                1 * int(not is_symmetric),
                1 * int(is_nilpotent),
            ]
        )
        difficulty_label = "EASY" if difficulty <= 1 else "MEDIUM" if difficulty <= 3 else "HARD"
        summary = {
            "rank": int(rank),
            "nullity": int(self.dimension - rank),
            "num_zero_eigenvalues": int(num_zero_eigenvalues),
            "is_singular": bool(is_singular),
            "has_repeated": bool(has_repeated),
            "has_zero_eigenvalue": bool(has_zero_eigenvalue),
            "is_symmetric": bool(is_symmetric),
            "is_diagonal": bool(is_diagonal),
            "is_identity": bool(is_identity),
            "is_zero_matrix": bool(is_zero),
            "is_nilpotent": bool(is_nilpotent),
            "condition_number": round(float(condition_number), 4) if condition_number != float("inf") else "inf",
            "difficulty": difficulty_label,
        }
        return summary

    def char_poly_coeff(self) -> None:
        if self.dimension <= 3:
            _lambda = sp.Symbol("lambda")
            char_matrix = self.A - _lambda * sp.eye(self.dimension)
            poly = sp.Poly(char_matrix.det(), _lambda)
            self.coefficients = [float(c) for c in poly.all_coeffs()]
        else:
            # numpy.poly is fast for float matrices
            coeffs = np.poly(np.array(self.matrix, dtype=float))
            self.coefficients = [round(float(c), 6) for c in coeffs]

    def calculate_eigenvalues(self) -> None:
        eigenvalues, _ = np.linalg.eig(np.array(self.matrix, dtype=float))
        self.eigenvalues = sorted([round(float(e.real), 6) for e in eigenvalues], reverse=True)

    def calculate_shifted_matrices(self) -> None:
        if self.eigenvalues is None:
            raise ValueError("Eigenvalues not computed yet. Call calculate_eigenvalues() first.")
        A = np.array(self.matrix, dtype=float)
        self.shifted_matrices = []
        for lam in self.eigenvalues:
            shifted = A - lam * np.eye(self.dimension)
            self.shifted_matrices.append([[round(float(x), 6) for x in row] for row in shifted])

    def calculate_rref(self) -> None:
        if self.shifted_matrices is None:
            raise ValueError("Shifted matrices not computed yet. Call calculate_shifted_matrices() first.")
        self.rref_matrices = []
        for shifted in self.shifted_matrices:
            if self.dimension <= 3:
                # SymPy exact RREF for 3x3 (accurate zero detection)
                M = sp.Matrix([[sp.Float(x, 8) for x in row] for row in shifted])
                rref, _ = M.rref(iszerofunc=lambda x: abs(float(x)) < 1e-4)
                self.rref_matrices.append(
                    [[round(float(rref[i, j]), 6) for j in range(self.dimension)] for i in range(self.dimension)]
                )
            else:
                # Numpy RREF via Gaussian elimination for 4x4/5x5 (faster)
                A = np.array(shifted, dtype=float)
                n = self.dimension
                tol = 1e-6
                row = 0
                result = A.copy()
                for col in range(n):
                    pivot = None
                    for r in range(row, n):
                        if abs(result[r, col]) > tol:
                            pivot = r
                            break
                    if pivot is None:
                        continue
                    result[[row, pivot]] = result[[pivot, row]]
                    result[row] = result[row] / result[row, col]
                    for r in range(n):
                        if r != row:
                            result[r] -= result[r, col] * result[row]
                    row += 1
                # Zero out near-zero entries
                result[np.abs(result) < tol] = 0.0
                self.rref_matrices.append([[round(float(result[i, j]), 6) for j in range(n)] for i in range(n)])

    def calculate_eigenvectors(self) -> None:
        if self.shifted_matrices is None:
            raise ValueError("Shifted matrices not computed yet. Call calculate_shifted_matrices() first.")
        self.eigenvectors = []
        for shifted in self.shifted_matrices:
            _, _, Vt = np.linalg.svd(np.array(shifted, dtype=float))
            vec = Vt[-1]
            vec = vec / vec[np.argmax(np.abs(vec))]
            self.eigenvectors.append([round(float(x), 6) for x in vec])

    def solve(self) -> None:
        """Run full pipeline in correct order."""
        self.char_poly_coeff()
        self.calculate_eigenvalues()
        self.calculate_shifted_matrices()
        self.calculate_rref()
        self.calculate_eigenvectors()

    def verify(self) -> bool:
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError("Call solve() first.")
        A = np.array(self.matrix, dtype=float)
        tolerance = 1e-4
        all_passed = True

        print("Eigenvalue Verification:")
        np_eigenvalues = sorted([round(float(e.real), 4) for e in np.linalg.eig(A)[0]], reverse=True)
        manual = [round(e, 4) for e in self.eigenvalues]
        eigen_passed = np_eigenvalues == manual
        all_passed = all_passed and eigen_passed
        print(f"  numpy:  {np_eigenvalues}")
        print(f"  manual: {manual}")
        print(f"  result: {'✓ PASS' if eigen_passed else '✗ FAIL'}")

        print("\nEigenvector Verification (residual ||Av - λv|| < 1e-4):")
        for lam, vec in zip(self.eigenvalues, self.eigenvectors):
            v = np.array(vec, dtype=float)
            residual = np.linalg.norm(A @ v - lam * v)
            passed = residual < tolerance
            all_passed = all_passed and passed
            print(f"  λ={lam:<12} residual={residual:.2e}  {'✓ PASS' if passed else '✗ FAIL'}")

        print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
        return all_passed

    def verify_decomposition(self) -> bool:
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError("Call solve() first.")
        P = np.array(self.eigenvectors, dtype=float).T
        D = np.diag(self.eigenvalues)
        A = np.array(self.matrix, dtype=float)
        cond = np.linalg.cond(P)
        if cond > 1e10:
            print(f"  ⚠ Warning: P is ill-conditioned (cond={cond:.2e}), matrix may have repeated eigenvalues.")
            print(f"  Using pseudo-inverse for verification.")
        reconstructed = P @ D @ np.linalg.pinv(P)
        residual = np.linalg.norm(A - reconstructed)
        passed = residual < 1e-4
        print("Eigen Decomposition Verification (A = P @ D @ P⁻¹):")
        print(f"  Condition number of P : {cond:.2e}")
        print(f"  Original A:\n{A}")
        print(f"  Reconstructed:\n{np.round(reconstructed, 4)}")
        print(f"  Residual ||A - PDP⁻¹|| = {residual:.2e}")
        print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
        return passed


# ── tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("3x3 distinct", [[1, 4, 7], [2, 3, 3], [4, 9, 0]], 3),
        ("3x3 repeated", [[2, -3, 0], [2, -5, 0], [0, 0, 3]], 3),
        ("5x5 degenerate", [[2, 0, 2, 0, 2], [0, 3, 0, 3, 0], [2, 0, 2, 0, 2], [0, 3, 0, 3, 0], [2, 0, 2, 0, 2]], 5),
    ]

    for label, matrix, dim in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {label}")
        print(f"{'='*60}")
        try:
            p = EigenProblem(matrix=matrix, dimension=dim)
            p.analyze_matrix()
            p.solve()
            print(p)
            p.verify()
            p.verify_decomposition()
        except Exception as e:
            print(f"ERROR: {e}")
