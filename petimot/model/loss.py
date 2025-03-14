import torch
from torch_geometric.data import Batch
from scipy.optimize import linear_sum_assignment
import sys


def compute_NSSE_matrix(
    eigvects: torch.Tensor,
    coverage: torch.Tensor,
    modes_pred: torch.Tensor,
) -> torch.Tensor:
    N, _, _ = eigvects.shape

    modes_pred = modes_pred - modes_pred.mean(dim=0, keepdim=True)

    # Expand tensors for broadcasting
    coverage = coverage.view(N, 1, 1, 1)
    sqrt_cov = torch.sqrt(coverage)  # (N, 1, 1, 1)
    eigvects_expanded = eigvects.unsqueeze(2)  # (N, gt, 1, 3)
    modes_pred_expanded = modes_pred.unsqueeze(1)  # (N, 1, pred, 3)

    # Optimal scaling calculation
    numerator = torch.sum(
        sqrt_cov * eigvects_expanded * modes_pred_expanded, dim=(0, -1)
    )
    denominator = torch.sum(coverage * modes_pred_expanded.pow(2), dim=(0, -1)) + 1e-8
    c_optimal = numerator / denominator

    modes_adjusted = (
        sqrt_cov * modes_pred_expanded * c_optimal.unsqueeze(0).unsqueeze(-1)
    )

    loss_matrix = (
        torch.sum((eigvects_expanded - modes_adjusted).pow(2), dim=(0, -1)) / N
    )  # nb_gt, nb_pred

    return loss_matrix


def select_minimum_indices(cost_matrix: torch.Tensor) -> torch.Tensor:

    cost_np = cost_matrix.detach().cpu().numpy()

    # Get optimal indices
    row_idx, col_idx = linear_sum_assignment(cost_np)

    indices = torch.stack(
        [
            torch.tensor(row_idx, device=cost_matrix.device),
            torch.tensor(col_idx, device=cost_matrix.device),
        ],
        dim=1,
    )

    return indices


def elementwise_loss(
    eigvects: torch.Tensor, coverage: torch.Tensor, modes_pred: torch.Tensor
) -> torch.Tensor:
    loss_matrix = compute_NSSE_matrix(eigvects, coverage, modes_pred)
    matched_indices = select_minimum_indices(loss_matrix)
    return loss_matrix[matched_indices[:, 0], matched_indices[:, 1]]  # LS loss


def compute_nsse_loss(batch: Batch, modes_pred: torch.Tensor) -> torch.Tensor:

    eigvects, coverage, ptr = batch.eigvects, batch.coverage, batch.ptr
    losses = []

    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i + 1]
        losses.append(
            elementwise_loss(
                eigvects[start:end],
                coverage[start:end],
                modes_pred[start:end],
            )
        )

    return torch.stack(losses)


def orthogonalize_modes(modes: torch.Tensor) -> torch.Tensor:
    N, K, D = modes.shape  # [number of residues, number of modes, 3]

    modes_reshaped = modes.transpose(0, 1).reshape(K, -1)  # shape [K, N*D]

    Q, _ = torch.linalg.qr(modes_reshaped.T)  # Q: [N*D, K], so that Q^T Q = I

    modes_ortho = Q.reshape(N, D, K).transpose(1, 2)  # [N, K, D]
    return modes_ortho  # this is Q, with Q^T Q = I


def compute_rmsip_sq(
    eigvects: torch.Tensor, coverage: torch.Tensor, modes_pred: torch.Tensor
) -> float:

    eigvects = eigvects - eigvects.mean(dim=0, keepdim=True)
    eigvects = eigvects / torch.norm(eigvects, dim=(0, 2), keepdim=True)

    modes_pred = modes_pred - modes_pred.mean(dim=0, keepdim=True)

    n_modes = min(eigvects.shape[1], modes_pred.shape[1])

    sqrt_cov = torch.sqrt(coverage)[:, None, None]  # shape [N, 1, 1]
    weighted_modes_pred = modes_pred * sqrt_cov  # shape [N, K, D]

    weighted_modes_pred_ortho = orthogonalize_modes(weighted_modes_pred)[:, :n_modes, :]

    eigvects = eigvects[:, :n_modes, :]

    inner_products = torch.einsum("nkd,nld->kl", eigvects, weighted_modes_pred_ortho)

    rmsip_sq = torch.sum(inner_products**2) / n_modes

    return rmsip_sq


def compute_rmsip(
    eigvects: torch.Tensor, coverage: torch.Tensor, modes_pred: torch.Tensor
) -> float:
    return torch.sqrt(compute_rmsip_sq(eigvects, coverage, modes_pred)).item()


def compute_rmsip_loss_sample(
    eigvects: torch.Tensor, coverage: torch.Tensor, modes_pred: torch.Tensor
) -> torch.Tensor:

    rmsip_sq = compute_rmsip_sq(eigvects, coverage, modes_pred)

    loss = 1.0 - rmsip_sq  # SS loss

    return loss


def compute_rmsip_loss(batch: Batch, modes_pred: torch.Tensor) -> torch.Tensor:

    eigvects, coverage, ptr = batch.eigvects, batch.coverage, batch.ptr
    losses = []

    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i + 1]
        loss = compute_rmsip_loss_sample(
            eigvects[start:end], coverage[start:end], modes_pred[start:end]
        )
        losses.append(loss)

    return torch.stack(losses)


def compute_rmsip_sq_without_ortho(
    eigvects: torch.Tensor, coverage: torch.Tensor, modes_pred: torch.Tensor
) -> float:

    eigvects = eigvects - eigvects.mean(dim=0, keepdim=True)
    eigvects = eigvects / torch.norm(eigvects, dim=(0, 2), keepdim=True)

    modes_pred = modes_pred - modes_pred.mean(dim=0, keepdim=True)

    n_modes = min(eigvects.shape[1], modes_pred.shape[1])

    sqrt_cov = torch.sqrt(coverage)[:, None, None]  # shape [N, 1, 1]
    weighted_modes_pred = modes_pred * sqrt_cov  # shape [N, K, D]
    weighted_modes_pred = weighted_modes_pred[:, :n_modes, :]
    weighted_modes_pred = weighted_modes_pred / torch.norm(
        weighted_modes_pred, dim=(0, 2), keepdim=True
    )

    eigvects = eigvects[:, :n_modes, :]

    inner_products = torch.einsum("nkd,nld->kl", eigvects, weighted_modes_pred)

    rmsip_sq = torch.sum(inner_products**2) / (n_modes * n_modes)

    return rmsip_sq


def compute_self_cosine_loss(
    modes_pred: torch.Tensor, coverage: torch.Tensor
) -> torch.Tensor:

    modes_pred = modes_pred - modes_pred.mean(dim=0, keepdim=True)

    coverage = coverage[:, None, None]  # [N, 1, 1]

    weighted_modes = modes_pred * torch.sqrt(coverage)  # [N, K, 3]

    norms = torch.norm(weighted_modes, dim=(0, 2), keepdim=True)  # [1, K, 1]
    normalized_modes = weighted_modes / norms  # [N, K, 3]

    cosine_matrix = torch.einsum("nid,njd->ij", normalized_modes, normalized_modes)

    # Mask out diagonal elements (we don't want to include self-similarity)
    mask = ~torch.eye(cosine_matrix.shape[0], dtype=bool, device=cosine_matrix.device)

    n_modes = cosine_matrix.shape[0]
    cosine_loss = torch.sum(cosine_matrix[mask] ** 2) / (n_modes * n_modes)

    return cosine_loss


def compute_combined_loss_sample(
    eigvects: torch.Tensor, coverage: torch.Tensor, modes_pred: torch.Tensor
) -> torch.Tensor:
    rmsip_sq = compute_rmsip_sq_without_ortho(eigvects, coverage, modes_pred)
    rmsip_loss = 1.0 - rmsip_sq

    cosine_loss = compute_self_cosine_loss(modes_pred, coverage)

    combined_loss = rmsip_loss + cosine_loss  # IS loss

    return combined_loss


def compute_ortho_loss(batch: Batch, modes_pred: torch.Tensor) -> torch.Tensor:
    eigvects, coverage, ptr = batch.eigvects, batch.coverage, batch.ptr
    losses = []

    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i + 1]
        loss = compute_combined_loss_sample(
            eigvects[start:end], coverage[start:end], modes_pred[start:end]
        )
        losses.append(loss)

    return torch.stack(losses)
