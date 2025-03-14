import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Tuple
from petimot.utils.rigid_utils import Rigid


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super(MLP, self).__init__()

        if num_layers <= 0:
            self.network = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]

            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_dim, output_dim))

            self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MPNNLayer(MessagePassing):

    def __init__(
        self,
        emb_dim: int,
        edge_dim: int,
        num_modes_pred: int,
        mlp_num_layers: int,
        dropout: float,
        normalize_between_layers: bool,
        center_between_layers: bool,
        orthogonalize_between_layers: bool,
        ablate_structure: bool,
    ):
        # We use `flow="target_to_source"`,
        # meaning the first row of edge_index are target nodes
        # and the second row are source nodes.
        super().__init__(aggr="mean", flow="target_to_source")
        self.normalize_between_layers = normalize_between_layers
        self.center_between_layers = center_between_layers
        self.orthogonalize_between_layers = orthogonalize_between_layers
        self.ablate_structure = ablate_structure

        if self.ablate_structure:
            message_input_dim = 2 * emb_dim + 3 * (num_modes_pred * 3)
        else:
            message_input_dim = 2 * emb_dim + edge_dim + 3 * (num_modes_pred * 3)
        # Message MLP => outputs new node embeddings (emb_dim)
        self.message_mlp = MLP(
            input_dim=message_input_dim,
            output_dim=emb_dim,
            hidden_dim=emb_dim,
            num_layers=mlp_num_layers,
            dropout=dropout,
        )

        # A linear layer to update v from [x, v]. (No bias if you prefer.)
        self.vect_mlp = nn.Linear(
            emb_dim + num_modes_pred * 3, num_modes_pred * 3, bias=False
        )

        self.num_modes_pred = num_modes_pred
        self.norm_x = nn.LayerNorm(emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rots: Rigid,
        edge_attr: torch.Tensor,
        ptr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_update = self.propagate(
            edge_index=edge_index,
            x=x,
            v=v,
            edge_rots=edge_rots,
            edge_attr=edge_attr,
        )

        x = x + x_update
        x = self.norm_x(x)

        v_update = self.vect_mlp(torch.cat([x, v], dim=-1))
        v = v + v_update

        if (
            self.normalize_between_layers
            or self.center_between_layers
            or self.orthogonalize_between_layers
        ):
            v = self.normalize_vectors(
                v,
                ptr,
                orthogonalize=self.orthogonalize_between_layers,
                normalize=self.normalize_between_layers,
                center=self.center_between_layers,
            )

        return x, v

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        v_i: torch.Tensor,
        v_j: torch.Tensor,
        edge_rots: Rigid,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:

        vj = v_j.reshape(-1, self.num_modes_pred, 3)
        vj_rotated = edge_rots.apply_batch(vj)
        vj_rotated = vj_rotated.view(vj_rotated.shape[0], -1)

        s1 = v_i - vj_rotated

        if self.ablate_structure:
            message = self.message_mlp(
                torch.cat([x_i, x_j, v_i, vj_rotated, s1], dim=-1)
            )
        else:
            message = self.message_mlp(
                torch.cat([x_i, x_j, v_i, vj_rotated, s1, edge_attr], dim=-1)
            )

        return message

    def normalize_vectors(
        self,
        v: torch.Tensor,
        ptr: torch.Tensor,
        orthogonalize: bool,
        normalize: bool,
        center: bool,
    ) -> torch.Tensor:

        v_norm = torch.empty_like(v)
        v_reshaped = v.view(
            v.shape[0], self.num_modes_pred, 3
        )  # [N, num_modes_pred, 3]

        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i + 1]
            n_residues = end - start
            v_graph = v_reshaped[start:end]

            if center:
                v_graph = v_graph - v_graph.mean(dim=0, keepdim=True)

            if orthogonalize and not normalize:
                norms_input = torch.norm(v_graph, dim=(0, 2))

            if orthogonalize:
                v_flat = v_graph.reshape(-1, self.num_modes_pred)

                q, _ = torch.linalg.qr(v_flat)

                if normalize:
                    # Scale Q to maintain magnitude proportional to sqrt(N)
                    scale = n_residues.to(v.device).sqrt()
                    q = q * scale
                else:
                    # Restore original norms
                    current_norms = torch.norm(q, dim=0)
                    scale_factors = norms_input / current_norms
                    q = q * scale_factors.unsqueeze(0)

                q_reshaped = q.reshape(n_residues, -1)
                v_norm[start:end] = q_reshaped

            else:
                if normalize:
                    # Just normalize without orthogonalization
                    scale = n_residues.to(v.device).sqrt()
                    current_norms = torch.norm(v_graph, dim=(0, 2))
                    scale_factors = scale / current_norms
                    v_graph = v_graph * scale_factors.unsqueeze(0).unsqueeze(-1)
                    v_norm[start:end] = v_graph.reshape(n_residues, -1)
                else:
                    # Neither normalize nor orthogonalize, just return (possibly centered) vectors
                    v_norm[start:end] = v_graph.reshape(n_residues, -1)

        return v_norm


class ProteinMotionMPNN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        edge_dim: int,
        num_modes_pred: int,
        num_layers: int,
        shared_layers: bool,
        mlp_num_layers: int,
        input_embedding_dropout: float,
        dropout: float,
        num_basis: int,
        max_dist: float,
        sigma: float,
        change_connectivity: bool,
        normalize_between_layers: bool,
        center_between_layers: bool,
        orthogonalize_between_layers: bool,
        start_with_zero_v: bool,
        ablate_structure: bool,
    ):
        super().__init__()

        self.num_modes_pred = num_modes_pred
        self.num_layers = num_layers
        self.shared_layers = shared_layers
        self.edge_dim = edge_dim

        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(input_embedding_dropout)

        self.num_basis = num_basis
        self.max_dist = max_dist
        self.sigma = sigma

        self.change_connectivity = change_connectivity
        self.start_with_zero_v = start_with_zero_v

        if shared_layers:
            self.mpnn = MPNNLayer(
                emb_dim=emb_dim,
                edge_dim=edge_dim,
                num_modes_pred=num_modes_pred,
                mlp_num_layers=mlp_num_layers,
                dropout=dropout,
                normalize_between_layers=normalize_between_layers,
                center_between_layers=center_between_layers,
                orthogonalize_between_layers=orthogonalize_between_layers,
                ablate_structure=ablate_structure,
            )
        else:
            self.mpnn_layers = nn.ModuleList()
            for _ in range(num_layers):
                layer = MPNNLayer(
                    emb_dim=emb_dim,
                    edge_dim=edge_dim,
                    num_modes_pred=num_modes_pred,
                    mlp_num_layers=mlp_num_layers,
                    dropout=dropout,
                    normalize_between_layers=normalize_between_layers,
                    center_between_layers=center_between_layers,
                    orthogonalize_between_layers=orthogonalize_between_layers,
                    ablate_structure=ablate_structure,
                )
                self.mpnn_layers.append(layer)

    def compute_radial_basis_fullbb(
        self, atoms_i: torch.Tensor, atoms_j: torch.Tensor
    ) -> torch.Tensor:

        # Pairwise distances among the sets of atoms
        # e.g. [E, 3, 3] for each edge => cdist => [E, 3, 3]
        distances = torch.cdist(atoms_i, atoms_j)  # shape [E, X, X]

        # Expand distances along a new dimension for the radial basis
        # => shape [E, X, X, num_basis]
        mu = torch.linspace(0, self.max_dist, self.num_basis, device=distances.device)
        radial = torch.exp(-((distances.unsqueeze(-1) - mu) ** 2) / (2 * self.sigma**2))

        # Flatten the X*X dimension
        # => shape [E, X*X*num_basis]
        radial = radial.view(distances.size(0), -1)
        return radial

    def compute_edge_features_from_index(
        self, rigids, row_index: torch.Tensor, col_index: torch.Tensor, bb: torch.Tensor
    ) -> Tuple:

        T_i = rigids[row_index]
        T_j = rigids[col_index]
        T_i_inv = T_i.invert()
        composed_ij = T_i_inv.compose(T_j)
        edge_rots = composed_ij.get_rots()
        edge_quats = edge_rots.get_quats()
        edge_trans = composed_ij.get_trans()
        edge_log_dist = torch.log(torch.norm(edge_trans, dim=-1) + 1e-8)
        chain_distance = torch.log(torch.abs(row_index - col_index) + 1)
        atoms_i = bb[row_index]
        atoms_j = bb[col_index]
        radial_features = self.compute_radial_basis_fullbb(atoms_i, atoms_j)
        edge_attr = torch.cat(
            [
                edge_quats,
                edge_trans,
                chain_distance.unsqueeze(-1),
                edge_log_dist.unsqueeze(-1),
                radial_features,
            ],
            dim=-1,
        )
        return edge_rots, edge_attr

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:

        emb = self.input_norm(data.x)
        emb = self.input_dropout(emb)

        emb = self.input_proj(emb)

        if self.start_with_zero_v:
            v = torch.zeros(emb.shape[0], self.num_modes_pred * 3, device=emb.device)
        else:
            v = torch.rand(emb.shape[0], self.num_modes_pred * 3, device=emb.device)

        row_index, col_index, ptr = data.row_index, data.col_index, data.ptr

        rigids = Rigid.make_transform_from_reference(
            data.bb[:, 0, :], data.bb[:, 1, :], data.bb[:, 2, :]
        )

        if self.change_connectivity:

            for layer_idx in range(self.num_layers):
                current_edge_index = torch.stack(
                    [row_index, col_index[layer_idx]], dim=0
                )

                edge_rots, edge_attr = self.compute_edge_features_from_index(
                    rigids, row_index, col_index[layer_idx], data.bb
                )

                if self.shared_layers:
                    emb, v = self.mpnn(
                        emb, v, current_edge_index, edge_rots, edge_attr, ptr
                    )
                else:
                    emb, v = self.mpnn_layers[layer_idx](
                        emb, v, current_edge_index, edge_rots, edge_attr, ptr
                    )
        else:
            edge_index = torch.stack([row_index, col_index], dim=0)
            edge_rots, edge_attr = self.compute_edge_features_from_index(
                rigids, row_index, col_index, data.bb
            )

            if self.shared_layers:
                for _ in range(self.num_layers):
                    emb, v = self.mpnn(emb, v, edge_index, edge_rots, edge_attr, ptr)
            else:
                for layer in self.mpnn_layers:
                    emb, v = layer(emb, v, edge_index, edge_rots, edge_attr, ptr)

        # Convert local vectors v to global coordinates
        v_reshaped = v.view(v.shape[0], self.num_modes_pred, 3)
        v_global = rigids.get_rots().apply_batch(v_reshaped)

        return emb, v_global
