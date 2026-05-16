import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_eigenvalues(pred: list[float], true: list[float]) -> tuple[list[float], list[float]]:
    """Match predicted eigenvalues to true eigenvalues using optimal assignment."""
    n = len(true)
    cost = np.abs(np.array(pred)[:, None] - np.array(true)[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pred = [pred[i] for i in row_ind]
    matched_true = [true[j] for j in col_ind]
    return matched_pred, matched_true


def eigenvalue_mae(pred: list[float], true: list[float]) -> float:
    """Mean absolute error between predicted and true eigenvalues."""
    matched_pred, matched_true = match_eigenvalues(pred, true)
    return float(np.mean(np.abs(np.array(matched_pred) - np.array(matched_true))))


def eigenvalue_accuracy(pred: list[float], true: list[float], tolerance: float = 0.0001) -> float:
    """Percentage of eigenvalues within tolerance of true values."""
    matched_pred, matched_true = match_eigenvalues(pred, true)
    correct = sum(1 for p, t in zip(matched_pred, matched_true) if abs(p - t) <= tolerance)
    return round(correct / len(true) * 100, 6)


def eigenvalue_mape(pred: list[float], true: list[float]) -> float:
    """Mean absolute percentage error for eigenvalues."""
    matched_pred, matched_true = match_eigenvalues(pred, true)
    errors = []
    for p, t in zip(matched_pred, matched_true):
        if abs(t) > 1e-10:
            errors.append(abs(p - t) / abs(t) * 100)
    return round(float(np.mean(errors)) if errors else 0.0, 6)


def eigenvector_cosine_similarity(pred: list[list[float]], true: list[list[float]]) -> float:
    """Average cosine similarity between predicted and true eigenvectors."""
    similarities = []
    for p, t in zip(pred, true):
        p_arr = np.array(p, dtype=float)
        t_arr = np.array(t, dtype=float)
        norm_p = np.linalg.norm(p_arr)
        norm_t = np.linalg.norm(t_arr)
        if norm_p < 1e-10 or norm_t < 1e-10:
            continue
        sim = abs(np.dot(p_arr, t_arr) / (norm_p * norm_t))
        similarities.append(sim)
    return round(float(np.mean(similarities)) if similarities else 0.0, 6)


def characteristic_polynomial_mae(pred: list[float], true: list[float]) -> float:
    """MAE between predicted and true characteristic polynomial coefficients."""
    if not pred or not true:
        return None
    min_len = min(len(pred), len(true))
    return round(float(np.mean(np.abs(np.array(pred[:min_len]) - np.array(true[:min_len])))), 6)


def get_intermediate_step(steps: list[dict], step_name: str):
    """Extract value of a specific step from intermediate_steps list."""
    for step in steps:
        if step["step"] == step_name:
            return step["value"]
    return None


def evaluate_sample(result: dict, ground_truth: dict) -> dict:
    """Compute all metrics for a single sample."""
    pred_eigenvalues = result["result"].get("eigenvalues", [])
    true_eigenvalues = ground_truth["result"].get("eigenvalues", [])
    pred_eigenvectors = result["result"].get("eigenvectors", [])
    true_eigenvectors = ground_truth["result"].get("eigenvectors", [])

    pred_poly = get_intermediate_step(result["intermediate_steps"], "characteristic_polynomial_coefficients")
    true_poly = get_intermediate_step(ground_truth["intermediate_steps"], "characteristic_polynomial_coefficients")

    metrics = {
        "id": result["id"],
        "eigenvalue_mae": eigenvalue_mae(pred_eigenvalues, true_eigenvalues),
        "eigenvalue_accuracy": eigenvalue_accuracy(pred_eigenvalues, true_eigenvalues, tolerance=0.0001),
        "eigenvalue_mape": eigenvalue_mape(pred_eigenvalues, true_eigenvalues),
        "eigenvector_cosine_similarity": eigenvector_cosine_similarity(pred_eigenvectors, true_eigenvectors),
        "characteristic_polynomial_mae": characteristic_polynomial_mae(pred_poly, true_poly),
    }
    return metrics


def evaluate_all(results_path: str, dataset_path: str, output_path: str = None):
    """Evaluate all results against ground truth dataset."""
    with open(results_path, "r") as f:
        results = json.load(f)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # index dataset by id for fast lookup
    gt_index = {sample["id"]: sample for sample in dataset}

    all_metrics = []
    for result in results:
        sample_id = result["id"]
        if sample_id not in gt_index:
            print(f"WARNING: {sample_id} not found in dataset, skipping")
            continue
        metrics = evaluate_sample(result, gt_index[sample_id])
        all_metrics.append(metrics)
        print(f"{sample_id}:")
        print(f"  Eigenvalue MAE:         {metrics['eigenvalue_mae']}")
        print(f"  Eigenvalue Accuracy:    {metrics['eigenvalue_accuracy']}%")
        print(f"  Eigenvalue MAPE:        {metrics['eigenvalue_mape']}%")
        print(f"  Cosine Similarity:      {metrics['eigenvector_cosine_similarity']}")
        print(f"  Char Poly MAE:          {metrics['characteristic_polynomial_mae']}")

    # overall summary
    print("\n--- OVERALL SUMMARY ---")
    for key in [
        "eigenvalue_mae",
        "eigenvalue_accuracy",
        "eigenvalue_mape",
        "eigenvector_cosine_similarity",
        "characteristic_polynomial_mae",
    ]:
        values = [m[key] for m in all_metrics if m[key] is not None]
        avg = round(float(np.mean(values)), 6) if values else None
        print(f"{key}: {avg}")

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to {output_path}")

    return all_metrics


if __name__ == "__main__":
    evaluate_all(
        results_path="results/result_llama4.json",
        dataset_path="data/dataset.json",
        output_path="results/m_result_llama4.json",
    )
