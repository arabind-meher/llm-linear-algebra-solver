import json


def prompt(matrix: list[list[float]], dimension: int) -> str:
    matrix_str = json.dumps(matrix)

    # Build dynamic intermediate steps structure based on dimension
    steps_example = [
        {"step": "characteristic_polynomial_coefficients", "value": [f"float" for _ in range(dimension + 1)]},
        {"step": "eigenvalues", "value": [f"float" for _ in range(dimension)]},
    ]

    for i in range(1, dimension + 1):
        steps_example.append(
            {
                "step": f"shifted_matrix_lambda_{i}",
                "value": [[f"float" for _ in range(dimension)] for _ in range(dimension)],
            }
        )
        steps_example.append(
            {"step": f"rref_lambda_{i}", "value": [[f"float" for _ in range(dimension)] for _ in range(dimension)]}
        )

    steps_str = json.dumps(steps_example, indent=4)

    return f"""
                You are a linear algebra expert. Solve the eigenvalue problem for this {dimension}x{dimension} matrix.

                Matrix: {matrix_str}

                Follow these steps exactly:
                1. Compute the characteristic polynomial coefficients of det(A - λI) = 0, highest degree first
                2. Solve for all {dimension} eigenvalues λ, sort them in descending order
                3. For each eigenvalue λ_i, compute the shifted matrix (A - λ_i * I), rounded to 6 decimal places
                4. Compute the RREF of each shifted matrix
                5. Compute eigenvectors from the RREF — normalize each so its largest absolute value component equals 1.0

                Respond ONLY in this exact JSON format with real float values, no explanation, no markdown:
                {{
                    "intermediate_steps": {steps_str},
                    "result": {{
                        "eigenvalues": [list of {dimension} floats sorted descending, rounded to 6 decimal places],
                        "eigenvectors": [
                            list of {dimension} eigenvectors each with {dimension} floats rounded to 6 decimal places,
                            normalized so the largest absolute value component in each eigenvector is 1.0,
                            order matches eigenvalue order
                        ]
                    }}
                }}

                Important: In the result section only, all floats must be rounded to a maximum of 6 decimal places.
            """
