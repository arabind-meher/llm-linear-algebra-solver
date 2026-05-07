# Improving LLM Reliability in Linear Algebra via Step-Based Execution

This project aims to improve the reliability of large language models (LLMs) in solving linear algebra problems involving multi-step computations with decimal inputs. While LLMs can explain concepts well, they often produce incorrect or imprecise numerical results. We propose a hybrid system where the LLM parses the problem into a structured format, and a custom Python-based step engine performs the computation by generating explicit intermediate steps. These verified steps are then used by the LLM to produce accurate, grounded explanations. We will evaluate this approach against a baseline LLM-only system to measure improvements in correctness, numerical precision, and reasoning consistency.

## Data

The data/ directory contains 55 eigenvalue and eigenvector problems across three matrix sizes (3×3, 4×4, 5×5) with decimal-valued entries. Each problem includes the input matrix, verified ground-truth intermediate steps (characteristic polynomial, shifted matrices, RREF), and final eigenvalues and eigenvectors. Problems are categorized by structural type (distinct, repeated, singular, symmetric positive definite, nilpotent) and difficulty level (EASY, MEDIUM, HARD).

See [`data/README.md`](data/README.md) for full details on the dataset structure, statistics, and record format.
