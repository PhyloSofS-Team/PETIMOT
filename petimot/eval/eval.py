import torch
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from petimot.model.loss import (
    compute_NSSE_matrix,
    compute_rmsip_sq,
    select_minimum_indices,
)


def compute_magnitude_error_matrix(
    eigvects: torch.Tensor, coverage: torch.Tensor, modes_pred: torch.Tensor
) -> torch.Tensor:
    N, nmode_gt, _ = eigvects.shape

    modes_pred = modes_pred - modes_pred.mean(dim=0, keepdim=True)

    eigvects = eigvects - eigvects.mean(dim=0, keepdim=True)

    gt_magnitudes = torch.norm(eigvects, dim=2)[:, :, None]  # Shape: (N, nmode_gt,1)
    pred_magnitudes = torch.norm(modes_pred, dim=2)[
        :, None, :
    ]  # Shape: (N, 1, nmode_pred)

    coverage = coverage[:, None, None]
    sqrt_cov = torch.sqrt(coverage)  # Shape: (N, 1, 1)

    numerator = torch.sum(sqrt_cov * gt_magnitudes * pred_magnitudes, dim=0)

    denominator = torch.sum(coverage * pred_magnitudes.pow(2), dim=0) + 1e-8
    c_optimal = numerator / denominator

    pred_magnitudes = (
        sqrt_cov * pred_magnitudes * c_optimal[None, :, :]
    )  # Shape: (N, 1, nmode_pred)

    sum_squared_error_matrix = (
        torch.sum((gt_magnitudes - pred_magnitudes) ** 2, dim=0) / N
    )

    return sum_squared_error_matrix


def compute_optimal_assignment_metrics(
    matrix: torch.Tensor, maximize: bool = False
) -> float:

    cost_matrix = -matrix if maximize else matrix

    # Get optimal indices using Hungarian algorithm
    indices = select_minimum_indices(cost_matrix)

    optimal_cost = matrix[indices[:, 0], indices[:, 1]].mean().item()

    return optimal_cost


def load_ground_truth(
    file_path: str, num_modes_gt: int, device: str
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        data = torch.load(file_path, map_location=device)
        eigvects = data["eigvects"]
        seq_length = int(len(eigvects) / 3)

        eigvects = eigvects[:, :num_modes_gt]
        eigvects *= seq_length**0.5
        eigvects = eigvects.reshape(-1, 3, num_modes_gt).permute(0, 2, 1)

        coverage = data.get("coverage", torch.ones(eigvects.shape[0], device=device))
        if not isinstance(coverage, torch.Tensor):
            coverage = torch.tensor(coverage, device=device)

        return eigvects, coverage

    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None


def load_predictions(
    base_path: str, base_name: str, num_modes: int, device: str
) -> Optional[torch.Tensor]:
    try:
        modes = []
        for k in range(num_modes):
            pred_file = os.path.join(base_path, f"{base_name}_mode_{k}.txt")
            modes.append(np.loadtxt(pred_file))
        return torch.tensor(np.stack(modes, axis=1), device=device, dtype=torch.float32)
    except Exception as e:
        print(f"Error loading predictions for {base_name}: {e}")
        return None


def save_matrix(
    output_path: str,
    base_name: str,
    matrix: torch.Tensor,
    num_modes_gt: int,
    metric_name: str,
):
    matrix_path = os.path.join(output_path, f"{base_name}_{metric_name}_matrix.csv")
    with open(matrix_path, "w") as f:
        f.write("mode_name," + ",".join(f"gt_{j}" for j in range(num_modes_gt)) + "\n")

        matrix_cpu = matrix.cpu()
        for i in range(len(matrix_cpu)):
            row = f"pred_{i}," + ",".join(f"{val:.6f}" for val in matrix_cpu[i])
            f.write(f"{row}\n")


def save_sample_metrics(output_path: str, base_name: str, metrics: Dict) -> None:
    metrics_path = os.path.join(output_path, f"{base_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def evaluate(
    prediction_path: str,
    ground_truth_path: str,
    output_path: str,
    sample_ids: List[str],
    num_modes_pred: int = 4,
    num_modes_gt: int = 4,
    device: str = "cuda",
    success_threshold: float = 0.6,
) -> Dict:

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    prediction_subdir = os.path.basename(prediction_path.rstrip("/"))
    output_path = os.path.join(output_path, prediction_subdir)

    os.makedirs(output_path, exist_ok=True)
    if not sample_ids:
        all_files = os.listdir(prediction_path)
        mode0_files = [f for f in all_files if f.endswith("_mode_0.txt")]
        if not mode0_files:
            raise ValueError(
                f"No files ending with '_mode_0.txt' found in {prediction_path}"
            )

        sample_ids = [Path(f).stem.rsplit("_mode", 1)[0] for f in mode0_files]

    sample_ids = [
        Path(sample_id).stem.rsplit("_mode", 1)[0] for sample_id in sample_ids
    ]

    min_losses = []
    min_magnitude_errors = []
    rmsip_sq_scores = []

    optimal_losses = []
    optimal_magnitudes = []
    stats = {"total": 0, "success": 0}

    missing_files = []
    for sample_id in sample_ids:
        mode0_file = os.path.join(prediction_path, f"{sample_id}_mode_0.txt")
        gt_file = os.path.join(ground_truth_path, f"{sample_id}.pt")

        if not os.path.exists(mode0_file):
            missing_files.append(f"Missing prediction file: {mode0_file}")
        if not os.path.exists(gt_file):
            missing_files.append(f"Missing ground truth file: {gt_file}")

    if missing_files:
        print("Warning: Some files are missing:")
        for msg in missing_files:
            print(f"  {msg}")
        print("Proceeding with available files...")

    for base_name in tqdm(sample_ids, desc="Evaluating samples"):
        gt_data = load_ground_truth(
            os.path.join(ground_truth_path, f"{base_name}.pt"), num_modes_gt, device
        )
        if gt_data is None:
            continue

        modes_pred = load_predictions(
            prediction_path, base_name, num_modes_pred, device
        )
        if modes_pred is None:
            continue

        eigvects, coverage = gt_data
        loss_matrix = compute_NSSE_matrix(eigvects, coverage, modes_pred).T
        magnitude_error_matrix = compute_magnitude_error_matrix(
            eigvects, coverage, modes_pred
        ).T

        rmsip_sq = compute_rmsip_sq(eigvects, coverage, modes_pred).item()

        optimal_loss = compute_optimal_assignment_metrics(loss_matrix, maximize=False)

        optimal_magnitude = compute_optimal_assignment_metrics(
            magnitude_error_matrix, maximize=False
        )

        stats["total"] += 1
        min_loss = torch.min(loss_matrix).item()
        min_magnitude_error = torch.min(magnitude_error_matrix).item()

        stats["success"] += int(min_loss < success_threshold)
        min_losses.append(min_loss)
        min_magnitude_errors.append(min_magnitude_error)
        rmsip_sq_scores.append(rmsip_sq)

        optimal_losses.append(optimal_loss)
        optimal_magnitudes.append(optimal_magnitude)

        sample_metrics = {
            "nsse_metrics": {"min_loss": min_loss, "optimal_assignment": optimal_loss},
            "magnitude_metrics": {
                "min_error": min_magnitude_error,
                "optimal_assignment": optimal_magnitude,
            },
            "rmsip_sq": rmsip_sq,
            "success": min_loss < success_threshold,
        }
        save_sample_metrics(output_path, base_name, sample_metrics)

        save_matrix(output_path, base_name, loss_matrix, num_modes_gt, "loss")
        save_matrix(
            output_path,
            base_name,
            magnitude_error_matrix,
            num_modes_gt,
            "magnitude_error",
        )

    results = {
        "total_samples": stats["total"],
        "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
        "nsse_metrics": {
            "mean_min_loss": float(np.mean(min_losses)) if min_losses else 0,
            "std_min_loss": float(np.std(min_losses)) if min_losses else 0,
            "optimal_assignment_mean": (
                float(np.mean(optimal_losses)) if optimal_losses else 0
            ),
            "optimal_assignment_std": (
                float(np.std(optimal_losses)) if optimal_losses else 0
            ),
        },
        "magnitude_metrics": {
            "mean_min_error": (
                float(np.mean(min_magnitude_errors)) if min_magnitude_errors else 0
            ),
            "std_min_error": (
                float(np.std(min_magnitude_errors)) if min_magnitude_errors else 0
            ),
            "optimal_assignment_mean": (
                float(np.mean(optimal_magnitudes)) if optimal_magnitudes else 0
            ),
            "optimal_assignment_std": (
                float(np.std(optimal_magnitudes)) if optimal_magnitudes else 0
            ),
        },
        "rmsip_sq_metrics": {
            "mean_rmsip_sq": float(np.mean(rmsip_sq_scores)) if rmsip_sq_scores else 0,
            "std_rmsip_sq": float(np.std(rmsip_sq_scores)) if rmsip_sq_scores else 0,
        },
    }

    with open(os.path.join(output_path, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {output_path}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"NSSE metrics:")
    print(
        f"  Min loss (mean ± std): {results['nsse_metrics']['mean_min_loss']:.4f} ± {results['nsse_metrics']['std_min_loss']:.4f}"
    )
    print(
        f"  Optimal assignment (mean ± std): {results['nsse_metrics']['optimal_assignment_mean']:.4f} ± {results['nsse_metrics']['optimal_assignment_std']:.4f}"
    )
    print(
        f"Mean rmsip_sq: {results['rmsip_sq_metrics']['mean_rmsip_sq']:.4f} ± {results['rmsip_sq_metrics']['std_rmsip_sq']:.4f}"
    )

    return results
