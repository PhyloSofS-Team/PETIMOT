from dataclasses import dataclass, field
import torch
import yaml


@dataclass
class ModelConfig:
    # Architecture parameters
    emb_dim: int = 256
    edge_dim: int = 329
    num_modes_pred: int = 4
    num_layers: int = 15
    shared_layers: bool = False
    mlp_num_layers: int = 1
    start_with_zero_v: bool = False

    # Regularization
    input_embedding_dropout: float = 0.8
    dropout: float = 0.4
    normalize_between_layers: bool = False
    center_between_layers: bool = False
    orthogonalize_between_layers: bool = False

    # Geometric features
    num_basis: int = 20
    num_backbone_atoms: int = 4
    max_dist: float = 20.0
    sigma: float = 1.0

    # Ablation
    ablate_structure: bool = False

    def __post_init__(self):
        expected_edge_dim = 9 + (self.num_backbone_atoms**2 * self.num_basis)
        if self.edge_dim != expected_edge_dim:
            raise ValueError(
                f"edge_dim must be 9 + (num_backbone_atoms² * num_basis) = "
                f"9 + ({self.num_backbone_atoms}² * {self.num_basis}) = {expected_edge_dim}. "
                f"Got {self.edge_dim}"
            )


@dataclass
class DataConfig:
    training_split_path: str = "full_train_list.txt"
    validation_split_path: str = "val_list.txt"
    ground_truth_dir: str = "ground_truth"
    embedding_dir: str = "embeddings"
    emb_model: str = "ProstT5"
    noise: float = 0.0
    k_nearest: int = 5
    l_random: int = 10
    num_modes_gt: int = 4
    rand_emb: bool = False
    change_connectivity: bool = True


@dataclass
class TrainingConfig:
    seed: int = 7
    nsse_weight: float = 0.5  # LS loss
    rmsip_weight: float = 0.5  # SS loss
    ortho_weight: float = 0.0  # IS loss
    batch_size: int = 32
    validation_batch_size: int = 32
    nb_epochs: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 10
    use_amp: bool = True
    scheduler_factor: float = 0.2
    scheduler_patience: int = 10
    early_stop_patience: int = 50
    loss_threshold: float = 0.6
    num_workers: int = 6
    num_workers_val: int = 2
    pin_memory: bool = False
    weights_dir: str = "weights"
    debug: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
        )

    def save(self, path: str) -> None:
        config_dict = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False)
