import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(values, mask):
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class LanguageModelEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(768)
        self.input_projection = nn.Linear(768, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, tokens, mask):
        values = self.input_projection(self.input_norm(tokens))
        values = self.encoder(values, src_key_padding_mask=~mask)
        values = self.output_norm(values)
        return values * mask.unsqueeze(-1).to(values.dtype)


class GraphMessageLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, normalized_adjacency, mask):
        messages = torch.bmm(normalized_adjacency, values)
        delta = self.update(torch.cat([values, messages], dim=-1))
        values = self.norm(values + self.dropout(delta))
        return values * mask.unsqueeze(-1).to(values.dtype)


class MolecularGraphEncoder(nn.Module):
    def __init__(self, d_model=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Linear(75, d_model)
        self.layers = nn.ModuleList([
            GraphMessageLayer(d_model, dropout) for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def normalize_adjacency(adjacency, mask):
        batch, length, _ = adjacency.shape
        identity = torch.eye(length, device=adjacency.device).unsqueeze(0).expand(batch, -1, -1)
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        adjacency = (adjacency + identity) * valid.to(adjacency.dtype)
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        return inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)

    def forward(self, atoms, adjacency, mask):
        values = self.input_projection(atoms)
        normalized_adjacency = self.normalize_adjacency(adjacency, mask)
        for layer in self.layers:
            values = layer(values, normalized_adjacency, mask)
        values = self.output_norm(values)
        return values * mask.unsqueeze(-1).to(values.dtype)


class PhysicochemicalPropertyEncoder(nn.Module):
    """Encode global molecular properties as one structural-stream token."""

    def __init__(self, input_dim=1613, d_model=128, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, values):
        return self.encoder(values)


class PhysicochemicalEvidenceIntegration(nn.Module):
    """Inject an order-invariant molecular-property pair into the GNN anchor."""

    def __init__(self, d_model=128, dropout=0.2):
        super().__init__()
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.layer_scale = nn.Parameter(torch.tensor(0.25))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, gnn_feature, molecule_a, molecule_b):
        pair = torch.cat([
            molecule_a + molecule_b,
            torch.abs(molecule_a - molecule_b),
            molecule_a * molecule_b,
        ], dim=-1)
        correction = self.pair_projection(pair)
        return self.output_norm(gnn_feature + self.layer_scale * correction)


class ExperimentalConditionModulatedAttention(nn.Module):
    """Cross-attention with a condition-dependent bilinear matching metric."""

    def __init__(self, d_model=128, n_heads=4, condition_dim=6, dropout=0.2):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.condition_q_scale = nn.Linear(condition_dim, d_model)
        self.condition_k_scale = nn.Linear(condition_dim, d_model)
        self.condition_v_scale = nn.Linear(condition_dim, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        for projection in (
            self.condition_q_scale, self.condition_k_scale, self.condition_v_scale
        ):
            nn.init.xavier_uniform_(projection.weight, gain=0.2)
            nn.init.constant_(projection.bias, 0.54132485)

    def _heads(self, values):
        batch, length, _ = values.shape
        return values.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, query_mask, key_mask, condition,
                use_condition):
        q = self._heads(self.q_projection(self.q_norm(query)))
        normalized_key = self.kv_norm(key)
        k = self._heads(self.k_projection(normalized_key))
        v = self._heads(self.v_projection(self.kv_norm(value)))
        if use_condition:
            def condition_scale(projection):
                scale = F.softplus(projection(condition)) + 1e-4
                return scale.view(len(query), self.n_heads, 1, self.head_dim)

            q = q * condition_scale(self.condition_q_scale)
            k = k * condition_scale(self.condition_k_scale)
            v = v * condition_scale(self.condition_v_scale)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~key_mask[:, None, None, :], -1e4)
        attention = self.attention_dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, v).transpose(1, 2).contiguous()
        output = output.view(len(query), query.shape[1], self.d_model)
        output = self.output_dropout(self.output_projection(output))
        return output * query_mask.unsqueeze(-1).to(output.dtype)


class CrossMolecularAwareness(nn.Module):
    """Shared bidirectional A<->B interaction for one modality."""

    def __init__(self, d_model=128, n_heads=4, condition_dim=6, dropout=0.2):
        super().__init__()
        self.cross_attention = ExperimentalConditionModulatedAttention(
            d_model, n_heads, condition_dim, dropout
        )
        self.attention_scale_logit = nn.Parameter(torch.tensor(0.0))
        self.condition_residual_gain = nn.Linear(condition_dim, 1)
        nn.init.xavier_uniform_(self.condition_residual_gain.weight, gain=0.2)
        nn.init.constant_(self.condition_residual_gain.bias, 0.54132485)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.ffn_scale = nn.Parameter(torch.tensor(0.1))

    def _update(self, query, source, query_mask, source_mask, condition,
                use_condition):
        delta = self.cross_attention(
            query, source, source, query_mask, source_mask, condition, use_condition
        )
        attention_scale = torch.sigmoid(self.attention_scale_logit)
        if use_condition:
            condition_gain = F.softplus(self.condition_residual_gain(condition)) + 1e-4
            delta = delta * condition_gain.unsqueeze(1)
        values = query + attention_scale * delta
        values = values + self.ffn_scale * self.ffn(self.ffn_norm(values))
        return values * query_mask.unsqueeze(-1).to(values.dtype)

    def forward(self, molecule_a, molecule_b, mask_a, mask_b, condition,
                use_condition):
        updated_a = self._update(
            molecule_a, molecule_b, mask_a, mask_b, condition, use_condition
        )
        updated_b = self._update(
            molecule_b, molecule_a, mask_b, mask_a, condition, use_condition
        )
        return updated_a, updated_b


class CrossMolecularSemanticAwareness(CrossMolecularAwareness):
    pass


class CrossMolecularStructuralAwareness(CrossMolecularAwareness):
    pass


class SymmetricInteractionPool(nn.Module):
    def __init__(self, d_model=128, dropout=0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, molecule_a, molecule_b, mask_a, mask_b):
        a = masked_mean(molecule_a, mask_a)
        b = masked_mean(molecule_b, mask_b)
        return self.projection(torch.cat([a + b, torch.abs(a - b), a * b], dim=-1))


class GNNAnchoredFusionGateProjection(nn.Module):
    """Learned intramolecular LM correction on an identity GNN feature path."""

    def __init__(self, d_model=128, n_heads=4, dropout=0.2):
        super().__init__()
        self.lm_norm = nn.LayerNorm(d_model)
        self.gnn_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.correction = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        # 0.5 * sigmoid(-1.386) = 0.10: a learnable GNN-favoring prior.
        self.residual_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, lm_feature, gnn_feature, lm_molecules, gnn_molecules,
                lm_masks, gnn_masks, **_):
        cross_deltas = []
        for molecule_index in range(2):
            lm_tokens = self.lm_norm(lm_molecules[molecule_index])
            gnn_tokens = self.gnn_norm(gnn_molecules[molecule_index])
            delta, _ = self.cross_attention(
                gnn_tokens, lm_tokens, lm_tokens,
                key_padding_mask=~lm_masks[molecule_index], need_weights=False,
            )
            cross_deltas.append(masked_mean(delta, gnn_masks[molecule_index]))
        cross_feature = torch.stack(cross_deltas, dim=0).mean(dim=0)
        correction = self.correction(torch.cat([
            lm_feature - gnn_feature, cross_feature
        ], dim=-1))
        residual_weight = 0.5 * torch.sigmoid(self.residual_logit)
        return self.output_norm(gnn_feature + residual_weight * correction)


class NSANet(nn.Module):
    def __init__(self, experiment, d_model=128, n_heads=4, gnn_layers=3,
                 dropout=0.2):
        super().__init__()
        self.experiment = experiment
        self.lm_encoder = LanguageModelEncoder(d_model, n_heads, dropout) if experiment.use_lm else None
        self.gnn_encoder = MolecularGraphEncoder(d_model, gnn_layers, dropout) if experiment.use_gnn else None
        self.physchem_encoder = (
            PhysicochemicalPropertyEncoder(1613, d_model, dropout)
            if experiment.use_physchem else None
        )
        self.physchem_pair_residual = (
            PhysicochemicalEvidenceIntegration(d_model, dropout)
            if experiment.physchem_mode == "pair_residual" else None
        )
        self.lm_pair_interaction = CrossMolecularSemanticAwareness(d_model, n_heads, 6, dropout) if experiment.use_lm else None
        self.gnn_pair_interaction = CrossMolecularStructuralAwareness(d_model, n_heads, 6, dropout) if experiment.use_gnn else None
        self.lm_pair_pool = SymmetricInteractionPool(d_model, dropout) if experiment.use_lm else None
        self.gnn_pair_pool = SymmetricInteractionPool(d_model, dropout) if experiment.use_gnn else None

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        if experiment.fusion == "gnn_anchor_cross_attention":
            self.fusion = GNNAnchoredFusionGateProjection(
                d_model, n_heads, dropout
            )
        elif experiment.fusion == "gnn_only":
            self.fusion = None
        else:
            raise ValueError(f"unsupported fusion: {experiment.fusion}")

    def _encode_semantic_pair(self, batch):
        semantic_states = [
            self.lm_encoder(item["tokens"], item["mask"]) for item in batch["lm"]
        ]
        masks = [item["mask"] for item in batch["lm"]]
        semantic_states[0], semantic_states[1] = self.lm_pair_interaction(
            semantic_states[0], semantic_states[1], masks[0], masks[1],
            batch["condition"], self.experiment.condition_lm,
        )
        semantic_pair_state = self.lm_pair_pool(
            semantic_states[0], semantic_states[1], masks[0], masks[1]
        )
        return semantic_pair_state, semantic_states, masks

    def _encode_compatibility_pair(self, batch):
        structural_states = [
            self.gnn_encoder(item["atoms"], item["adjacency"], item["mask"])
            for item in batch["graphs"]
        ]
        masks = [item["mask"] for item in batch["graphs"]]
        if batch["physchem"] is None:
            raise ValueError("physicochemical features are required")
        physicochemical_states = [
            self.physchem_encoder(batch["physchem"][molecule_index])
            for molecule_index in range(2)
        ]
        structural_states[0], structural_states[1] = self.gnn_pair_interaction(
            structural_states[0], structural_states[1], masks[0], masks[1],
            batch["condition"], self.experiment.condition_gnn,
        )
        topology_pair_state = self.gnn_pair_pool(
            structural_states[0], structural_states[1], masks[0], masks[1]
        )
        compatibility_pair_state = self.physchem_pair_residual(
            topology_pair_state, physicochemical_states[0],
            physicochemical_states[1]
        )
        return compatibility_pair_state, structural_states, masks

    def forward(self, batch):
        semantic_output = (
            self._encode_semantic_pair(batch) if self.experiment.use_lm else None
        )
        compatibility_output = (
            self._encode_compatibility_pair(batch)
            if self.experiment.use_gnn else None
        )
        if semantic_output is None:
            multimodal_pair_state = compatibility_output[0]
        elif compatibility_output is None:
            multimodal_pair_state = semantic_output[0]
        else:
            multimodal_pair_state = self.fusion(
                lm_feature=semantic_output[0],
                gnn_feature=compatibility_output[0],
                lm_molecules=semantic_output[1],
                gnn_molecules=compatibility_output[1],
                lm_masks=semantic_output[2],
                gnn_masks=compatibility_output[2],
            )
        return {
            "logits": self.classifier(multimodal_pair_state),
            "fused_feature": multimodal_pair_state,
            "lm_feature": semantic_output[0] if semantic_output is not None else None,
            "gnn_feature": (
                compatibility_output[0] if compatibility_output is not None else None
            ),
        }
