import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "seeds"
OUTPUT = ROOT / "Experiment.csv"


CORE_METRICS = [
    "positive_prevalence", "majority_accuracy_baseline", "no_skill_ap_baseline",
    "roc_auc", "partial_auc_fpr_0_10", "average_precision", "accuracy",
    "balanced_accuracy", "precision", "recall", "specificity", "npv",
    "f1", "f2", "mcc", "brier", "ece_10bin",
    "precision_at_10pct", "recall_at_10pct", "enrichment_factor_at_10pct",
    "precision_at_20pct", "recall_at_20pct", "enrichment_factor_at_20pct",
    "condition_shuffle_mean_abs_probability_delta",
    "condition_shuffle_roc_auc", "condition_shuffle_roc_auc_drop",
    "condition_shuffle_average_precision", "condition_shuffle_average_precision_drop",
]


def main():
    seed_rows = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        components = result["components"]
        row = {
            "row_type": "seed",
            "experiment_id": result["experiment_id"],
            "model_name": result["model_name"],
            "architecture_version": result.get("architecture_version", "legacy_unversioned"),
            "seed": result["seed"],
            "lm_encoder": components["lm_encoder"],
            "gnn_encoder": components["gnn_encoder"],
            "condition_lm": components["condition_lm"],
            "condition_gnn": components["condition_gnn"],
            "fusion": components["fusion"],
            "contrastive_learning": False,
            "domain_alignment": False,
            "descriptor_classifier": False,
            "parameter_count": result["parameter_count"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            "runtime_seconds": result["runtime_seconds"],
            "source_file": str(path.relative_to(ROOT)),
        }
        row.update({metric: result["validation"][metric] for metric in CORE_METRICS})
        seed_rows.append(row)

    rows = []
    for experiment_id in sorted({row["experiment_id"] for row in seed_rows}):
        group = [row for row in seed_rows if row["experiment_id"] == experiment_id]
        for row in group:
            for metric in CORE_METRICS:
                values = np.asarray([item[metric] for item in group], dtype=float)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"best_{metric}"] = (float(values.min()) if metric in {"brier", "ece_10bin"}
                                          else float(values.max()))
            rows.append(row)
        summary = dict(group[0])
        summary["row_type"] = "summary"
        summary["seed"] = "ALL"
        summary["source_file"] = f"{len(group)} seed rows"
        for metric in CORE_METRICS:
            values = np.asarray([item[metric] for item in group], dtype=float)
            summary[metric] = float(values.mean())
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"best_{metric}"] = (float(values.min()) if metric in {"brier", "ece_10bin"}
                                         else float(values.max()))
        rows.append(summary)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no seed results found")
        return
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
