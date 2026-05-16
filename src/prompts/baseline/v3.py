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
                1. Compute the characteristic polynomial coefficients of det(A - λI) = 0, highest degree first.
                Important: the leading coefficient must be {'-1.0' if dimension % 2 == 1 else '1.0'} for a {dimension}x{dimension} matrix.
                Double-check your polynomial before solving for eigenvalues.

                2. Solve for all {dimension} eigenvalues λ from the characteristic polynomial.
                Sort eigenvalues in descending order.

                3. For each eigenvalue λ_i, compute the shifted matrix (A - λ_i * I), rounded to 6 decimal places.

                4. Compute the RREF of each shifted matrix.

                5. Compute eigenvectors from the RREF.
                Normalize each eigenvector so the component with the largest absolute value equals exactly 1.0.
                Example: if raw vector is [2.0, -4.0, 1.5], largest abs value is 4.0, divide all by 4.0 → [0.5, -1.0, 0.375]
                Order eigenvectors to match eigenvalue order.

                Respond ONLY in this exact JSON format with real float values, no explanation, no markdown:
                {{
                    "intermediate_steps": {steps_str},
                    "result": {{
                        "eigenvalues": [list of {dimension} floats sorted descending, rounded to 6 decimal places],
                        "eigenvectors": [
                            list of {dimension} eigenvectors each with {dimension} floats rounded to 6 decimal places,
                            normalized so the largest absolute value component in each eigenvector is exactly 1.0,
                            order matches eigenvalue order
                        ]
                    }}
                }}

                Important: In the result section only, all floats must be rounded to a maximum of 6 decimal places.
            """
