from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    model_name: str
    use_lm: bool
    use_gnn: bool
    condition_lm: bool
    condition_gnn: bool
    fusion: str
    description: str
    use_physchem: bool = False
    physchem_mode: str = "none"


def _cfg(experiment_id, model_name, use_lm, use_gnn, condition_lm,
         condition_gnn, fusion, description, use_physchem=False,
         physchem_mode="none"):
    return ExperimentConfig(
        experiment_id, model_name, use_lm, use_gnn, condition_lm,
        condition_gnn, fusion, description, use_physchem, physchem_mode
    )


EXPERIMENTS = {
    "NSA-Net-C": _cfg(
        "NSA-Net-C", "NSA-Net-C", True, True, False, True,
        "gnn_anchor_cross_attention",
        "Condition-aware NSA-Net with symmetric Mordred pair residual",
        use_physchem=True, physchem_mode="pair_residual",
    ),
    "NSA-Net-S": _cfg(
        "NSA-Net-S", "NSA-Net-S", True, True, False, False,
        "gnn_anchor_cross_attention",
        "Condition-free NSA-Net with symmetric Mordred pair residual",
        use_physchem=True, physchem_mode="pair_residual",
    ),
    "NSA-Net-G": _cfg(
        "NSA-Net-G", "NSA-Net-G", False, True, False, False,
        "gnn_only",
        "Graph-physicochemical NSA-Net without semantic or condition learning",
        use_physchem=True, physchem_mode="pair_residual",
    ),
}


SEEDS = [42, 123, 2023, 2024, 3407]
