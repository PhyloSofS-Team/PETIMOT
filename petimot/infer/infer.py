import torch
import logging
import random
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

from torch_geometric.loader import DataLoader

from petimot.model.neural_net import ProteinMotionMPNN
from petimot.config.config import Config
from petimot.data.data_set import InferenceDataset
from petimot.utils.seeding import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EMBEDDING_DIM_MAP = {"prostt5": 1024, "esmc_300m": 960, "esmc_600m": 1152}


def setup_model(model_path: str, config: Config, device: str) -> ProteinMotionMPNN:

    model = ProteinMotionMPNN(
        input_dim=EMBEDDING_DIM_MAP[config.data.emb_model.lower()],
        emb_dim=config.model.emb_dim,
        edge_dim=config.model.edge_dim,
        num_modes_pred=config.model.num_modes_pred,
        num_layers=config.model.num_layers,
        shared_layers=config.model.shared_layers,
        mlp_num_layers=config.model.mlp_num_layers,
        input_embedding_dropout=0.0,
        dropout=0.0,
        num_basis=config.model.num_basis,
        max_dist=config.model.max_dist,
        sigma=config.model.sigma,
        change_connectivity=config.data.change_connectivity,
        normalize_between_layers=config.model.normalize_between_layers,
        center_between_layers=config.model.center_between_layers,
        orthogonalize_between_layers=config.model.orthogonalize_between_layers,
        start_with_zero_v=config.model.start_with_zero_v,
        ablate_structure=config.model.ablate_structure,
    ).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        error_msg = f"Failed to load model weights from {model_path}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return model


def save_predictions(output_dir: Path, name: str, modes_pred: torch.Tensor) -> None:

    modes_pred = modes_pred - modes_pred.mean(dim=0, keepdim=True)
    modes_pred_np = modes_pred.cpu().numpy()
    num_modes = modes_pred_np.shape[1]

    for k in range(num_modes):
        out_path = output_dir / f"{name}_mode_{k}.txt"
        try:
            np.savetxt(out_path, modes_pred_np[:, k, :], fmt="%.6f")
            logger.debug(f"Saved mode {k} predictions to {out_path}")
        except Exception as e:
            logger.error(f"Failed to save prediction to {out_path}: {e}")


def infer(
    model_path: str,
    config_file: str,
    input_list: List[str],
    output_path: str,
    device: str = "cuda",
) -> None:

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_stem = Path(model_path).stem
    output_subdir = output_dir / model_stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    config = Config.from_file(config_file)
    set_seed(config.training.seed, deterministic_algorithms=True)

    model = setup_model(model_path, config, device)

    dataset = InferenceDataset(
        entries=input_list,
        ground_truth_dir=config.data.ground_truth_dir,
        embedding_dir=config.data.embedding_dir,
        emb_model=config.data.emb_model,
        device=device,
        k_nearest=config.data.k_nearest,
        l_random=config.data.l_random,
        num_layers=config.model.num_layers,
        change_connectivity=config.data.change_connectivity,
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config.training.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.validation_batch_size,
        num_workers=config.training.num_workers_val,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
    )

    logger.info(f"Starting inference on {len(dataset)} protein(s)")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing proteins"):
            try:
                batch = batch.to(device)

                _, modes_pred = model(batch)  # (N, emb_dim), (N, num_modes, 3)

                ptr = batch.ptr  # shape [num_graphs + 1]
                for graph_idx, (start_idx, end_idx) in enumerate(
                    zip(ptr[:-1], ptr[1:])
                ):
                    name = batch.name[graph_idx]
                    modes_pred_graph = modes_pred[start_idx:end_idx]

                    save_predictions(output_subdir, name, modes_pred_graph)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

    logger.info(f"Inference complete. Results saved to {output_subdir}")
