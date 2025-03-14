import torch
from torch_geometric.data import Data, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from pathlib import Path
import logging
from .embeddings import EmbeddingManager
from .pdb_utils import load_backbone_coordinates
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_edges_no_self(num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    mask = i != j
    return torch.vstack((i[mask], j[mask]))


def build_knn_edges(
    dist_matrix: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = dist_matrix.size(0)
    _, nearest_neighbors = torch.topk(dist_matrix, k=k + 1, largest=False)
    nearest_neighbors = nearest_neighbors[:, 1:]
    k = nearest_neighbors.size(1)
    row_indices = (
        torch.arange(num_nodes, device=dist_matrix.device)[:, None]
        .expand(-1, k)
        .flatten()
    )
    col_indices = nearest_neighbors.flatten()
    mask = torch.ones(
        (num_nodes, num_nodes), dtype=torch.bool, device=dist_matrix.device
    )
    mask[torch.arange(num_nodes), torch.arange(num_nodes)] = False
    mask.scatter_(1, nearest_neighbors, False)

    return row_indices, col_indices, mask


def build_random_edges(
    num_nodes: int, mask: torch.Tensor, l: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rand_probs = torch.rand((num_nodes, num_nodes), device=mask.device).masked_fill(
        ~mask, -float("inf")
    )
    _, random_indices = torch.topk(rand_probs, l, dim=1)
    row_indices = (
        torch.arange(num_nodes, device=mask.device)[:, None].expand(-1, l).flatten()
    )
    col_indices = random_indices.flatten()

    return row_indices, col_indices


def build_connectivity(
    seq_len: int,
    dist_matrix: torch.Tensor,
    k_nearest: int,
    l_random: int,
    num_layers: int,
    change_connectivity: bool,
) -> tuple[torch.Tensor, torch.Tensor]:

    if k_nearest + l_random >= seq_len - 1:
        row_index, col_index_fixed = generate_edges_no_self(seq_len)
        if change_connectivity:
            col_index = col_index_fixed.unsqueeze(0).repeat(num_layers, 1)
        else:
            col_index = col_index_fixed
        return row_index, col_index

    if k_nearest > 0:
        row_index_knn, col_index_knn, mask = build_knn_edges(dist_matrix, k_nearest)
    else:

        mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        mask.fill_diagonal_(False)
        row_index_knn = torch.empty(0, dtype=torch.long)
        col_index_knn = torch.empty(0, dtype=torch.long)

    if not change_connectivity:
        if l_random > 0:
            row_index_rand, col_index_rand = build_random_edges(seq_len, mask, l_random)
        else:
            row_index_rand = torch.empty(0, dtype=torch.long)
            col_index_rand = torch.empty(0, dtype=torch.long)
        row_index_total = torch.cat([row_index_knn, row_index_rand], dim=0)
        col_index_total = torch.cat([col_index_knn, col_index_rand], dim=0)
        return row_index_total, col_index_total
    else:
        fixed_row = row_index_knn
        fixed_col = col_index_knn
        col_indices_layers = []
        for _ in range(num_layers):
            if l_random > 0:
                row_index_rand_layer, col_index_rand_layer = build_random_edges(
                    seq_len, mask, l_random
                )
            else:
                row_index_rand_layer = torch.empty(0, dtype=torch.long)
                col_index_rand_layer = torch.empty(0, dtype=torch.long)
            combined_col = torch.cat([fixed_col, col_index_rand_layer], dim=0)
            col_indices_layers.append(combined_col)

        if l_random > 0:
            row_index_rand = row_index_rand_layer
        else:
            row_index_rand = torch.empty(0, dtype=torch.long)
        total_row_index = torch.cat([fixed_row, row_index_rand], dim=0)
        col_index = torch.stack(col_indices_layers, dim=0)
        return total_row_index, col_index


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        ground_truth_dir: Union[str, Path],
        embedding_dir: Union[str, Path],
        emb_model: str,
        device: str,
        k_nearest: int,
        l_random: int,
        num_layers: int,
        change_connectivity: bool,
    ):
        super().__init__()
        self.ground_truth_dir = Path(ground_truth_dir)
        self.embedding_dir = Path(embedding_dir)
        self.emb_model = emb_model.lower()
        self.device = device
        self.k_nearest = k_nearest
        self.l_random = l_random
        self.num_layers = num_layers
        self.change_connectivity = change_connectivity

    def _build_connectivity(
        self, ca_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist_matrix = torch.cdist(ca_coords, ca_coords)
        return build_connectivity(
            seq_len=ca_coords.size(0),
            dist_matrix=dist_matrix,
            k_nearest=self.k_nearest,
            l_random=self.l_random,
            num_layers=self.num_layers,
            change_connectivity=self.change_connectivity,
        )

    def _load_embedding(self, name: str, seq_len: int) -> torch.Tensor:
        emb_path = self.embedding_dir / f"{name}_{self.emb_model}.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding {emb_path} not found")

        emb = torch.load(emb_path, map_location="cpu", weights_only=True)
        if emb.size(0) == seq_len + 2:  # Handle BOS/EOS tokens
            emb = emb[1:-1]
        elif emb.size(0) != seq_len:
            raise ValueError(f"Embedding size mismatch: {emb.size(0)} vs {seq_len}")

        return emb

    def _compute_embeddings(self, names: List[str], sequences: List[str]) -> None:
        """Shared logic for computing missing embeddings."""
        if not names:
            return

        logger.info(f"Computing embeddings for {len(names)} samples")
        EmbeddingManager(
            embedding_dir=self.embedding_dir,
            emb_model=self.emb_model,
            device=self.device,
        ).get_or_compute_embeddings(names, sequences)

    @abstractmethod
    def __getitem__(self, idx: int) -> Data:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class InferenceDataset(BaseDataset):
    def __init__(
        self,
        entries: List[str],
        ground_truth_dir: Union[str, Path],
        embedding_dir: Union[str, Path],
        emb_model: str,
        device: str,
        k_nearest: int,
        l_random: int,
        num_layers: int,
        change_connectivity: bool,
    ):
        super().__init__(
            ground_truth_dir=ground_truth_dir,
            embedding_dir=embedding_dir,
            emb_model=emb_model,
            device=device,
            k_nearest=k_nearest,
            l_random=l_random,
            num_layers=num_layers,
            change_connectivity=change_connectivity,
        )
        self.file_paths = self._resolve_entries(entries)
        self._compute_missing_embeddings()

    def _resolve_entries(self, entries: List[str]) -> List[Path]:
        resolved = []
        for entry in entries:
            path = Path(entry)
            if path.exists():
                resolved.append(path.resolve())
                continue

            for ext in [".pt"]:
                candidate = self.ground_truth_dir / f"{entry}{ext}"
                if candidate.exists():
                    resolved.append(candidate)
                    break
            else:
                logger.warning(f"Couldn't resolve entry: {entry}")

        if not resolved:
            raise ValueError("No valid input files found")
        return resolved

    def _compute_missing_embeddings(self) -> None:
        valid_names, sequences = [], []
        for path in self.file_paths:
            emb_path = self.embedding_dir / f"{path.stem}_{self.emb_model}.pt"
            if emb_path.exists():
                continue

            try:
                sequences.append(self._extract_sequence(path))
                valid_names.append(path.stem)
            except Exception as e:
                logger.error(f"Failed processing {path.name}: {e}")

        self._compute_embeddings(valid_names, sequences)

    def _extract_sequence(self, path: Path) -> str:
        if path.suffix == ".pt":
            return torch.load(path, map_location="cpu", weights_only=True).get(
                "seq", ""
            )
        elif path.suffix == ".pdb":
            seq = load_backbone_coordinates(str(path), allow_hetatm=True)["seq"]
            return seq
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Data:
        path = self.file_paths[idx]

        if path.suffix == ".pt":
            data = torch.load(path, map_location="cpu", weights_only=True)
            bb, coverage = data["bb"], data.get("coverage", torch.ones(len(data["bb"])))
        else:
            bb = load_backbone_coordinates(str(path), allow_hetatm=True)["bb"]
            coverage = torch.ones(len(bb))

        row_idx, col_idx = self._build_connectivity(bb[:, 1])

        return Data(
            x=self._load_embedding(path.stem, bb.size(0)),
            row_index=row_idx,
            col_index=col_idx,
            bb=bb,
            coverage=coverage,
            name=path.stem,
            path=str(path),
        )
