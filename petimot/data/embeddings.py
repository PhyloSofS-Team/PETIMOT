from pathlib import Path
import torch
import re
from transformers import T5Tokenizer, T5EncoderModel
from typing import List, Union


class EmbeddingManager:
    def __init__(
        self,
        embedding_dir: Path,
        emb_model: str,
        device: Union[str, torch.device] = "cuda",
    ):
        emb_model = emb_model.lower()
        valid_models = ["prostt5", "esmc_300m", "esmc_600m"]
        if emb_model not in valid_models:
            raise ValueError(
                f"Unsupported model: {emb_model}. Choose from {valid_models}."
            )

        self.embedding_dir = Path(embedding_dir)
        self.embedding_dir.mkdir(exist_ok=True)
        self.emb_model = emb_model

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if emb_model == "prostt5":
            self.tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/ProstT5", do_lower_case=False, legacy=True
            )
            self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(
                self.device
            )

            if self.device.type == "cuda":
                self.model = self.model.half()
            else:
                self.model = self.model.float()

        elif emb_model.startswith("esmc"):
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig

            self.model = ESMC.from_pretrained(emb_model).to(self.device)
            self.model = self.model.float()
            self.ESMProtein = ESMProtein
            self.LogitsConfig = LogitsConfig
            self.tokenizer = None

        self.model.eval()

    def get_or_compute_embeddings(self, names: List[str], sequences: List[str]) -> None:
        for name, seq in zip(names, sequences):
            emb_path = self.embedding_dir / f"{name}_{self.emb_model}.pt"
            if not emb_path.exists():
                emb = self._compute_embedding(seq)
                torch.save(emb.cpu(), emb_path)

    @torch.no_grad()
    def _compute_embedding(self, sequence: str) -> torch.Tensor:
        sequence = re.sub(r"[UZOB]", "X", sequence)

        if self.emb_model == "prostt5":
            sequence = "<AA2fold> " + " ".join(list(sequence))
            ids = self.tokenizer.encode_plus(
                sequence, add_special_tokens=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(ids.input_ids, attention_mask=ids.attention_mask)
            embeddings = outputs.last_hidden_state.squeeze(0)[1:-1].float()

        elif self.emb_model.startswith("esmc"):
            protein = self.ESMProtein(sequence=sequence)
            protein_tensor = self.model.encode(protein)
            logits_output = self.model.logits(
                protein_tensor, self.LogitsConfig(sequence=True, return_embeddings=True)
            )

            embeddings = logits_output.embeddings.squeeze(0)

        return embeddings
