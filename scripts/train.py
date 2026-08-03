import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from nsanet.config import EXPERIMENTS, SEEDS
from nsanet.data import create_dataloaders, move_batch
from nsanet.metrics import compute_metrics
from nsanet.model import NSANet


HERE = ROOT


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_digest(split_dir):
    split_dir = Path(split_dir)
    smiles_a = np.load(split_dir / "mol_0_smiles.npy", allow_pickle=True).astype(str)
    smiles_b = np.load(split_dir / "mol_1_smiles.npy", allow_pickle=True).astype(str)
    condition = np.asarray(
        np.load(split_dir / "descriptors.npy", allow_pickle=True), dtype=np.float32
    )
    raw_labels = np.load(split_dir / "labels.npy", allow_pickle=True)
    labels = np.asarray([int(np.asarray(value).reshape(-1)[0]) for value in raw_labels])
    digest = hashlib.sha256()
    for index in range(len(labels)):
        digest.update(smiles_a[index].encode("utf-8"))
        digest.update(b"\0")
        digest.update(smiles_b[index].encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(condition[index], dtype="<f4").tobytes())
        digest.update(np.asarray([labels[index]], dtype="<i8").tobytes())
    return digest.hexdigest()


def class_weights(train_loader, device):
    labels = np.asarray([
        int(np.asarray(value).reshape(-1)[0]) for value in train_loader.dataset.labels
    ])
    counts = np.bincount(labels, minlength=2)
    weights = len(labels) / (2.0 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.tolist()


def learning_rate_factor(epoch, epochs, warmup_epochs):
    warmup = min(1.0, epoch / max(warmup_epochs, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / epochs))
    return warmup * (0.05 + 0.95 * cosine)


def train_epoch(model, loader, optimizer, loss_weights, device):
    model.train()
    losses, labels, predictions = [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = F.cross_entropy(output["logits"], batch["labels"], weight=loss_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.extend(batch["labels"].detach().cpu().tolist())
        predictions.extend(output["logits"].argmax(dim=1).detach().cpu().tolist())
    return {
        "train_loss": float(np.mean(losses)),
        "train_accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))),
        "train_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


@torch.no_grad()
def evaluate_once(model, loader, device):
    model.eval()
    labels, scores, condition_shuffled_scores = [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model(batch)["logits"]
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        shuffled_batch = dict(batch)
        shuffled_batch["condition"] = torch.roll(batch["condition"], shifts=1, dims=0)
        shuffled_logits = model(shuffled_batch)["logits"]
        shuffled_probabilities = torch.softmax(shuffled_logits.float(), dim=1)[:, 1]
        labels.extend(batch["labels"].cpu().tolist())
        scores.extend(probabilities.cpu().tolist())
        condition_shuffled_scores.extend(shuffled_probabilities.cpu().tolist())
    metrics = compute_metrics(labels, scores, threshold=0.5)
    shuffled_metrics = compute_metrics(labels, condition_shuffled_scores, threshold=0.5)
    metrics.update({
        "condition_shuffle_mean_abs_probability_delta": float(np.mean(np.abs(
            np.asarray(scores) - np.asarray(condition_shuffled_scores)
        ))),
        "condition_shuffle_roc_auc": shuffled_metrics["roc_auc"],
        "condition_shuffle_roc_auc_drop": (
            metrics["roc_auc"] - shuffled_metrics["roc_auc"]
        ),
        "condition_shuffle_average_precision": shuffled_metrics["average_precision"],
        "condition_shuffle_average_precision_drop": (
            metrics["average_precision"] - shuffled_metrics["average_precision"]
        ),
    })
    return metrics, np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def train_seed(experiment, seed, args, device):
    set_seed(seed)
    (train_loader, validation_loader, descriptor_mean, descriptor_std,
     physchem_mean, physchem_std) = create_dataloaders(
         args.dataset_root, args.batch_size, seed
     )
    model = NSANet(
        experiment, d_model=args.d_model, n_heads=args.n_heads,
        gnn_layers=args.gnn_layers, dropout=args.dropout,
    ).to(device)
    weights, counts = class_weights(train_loader, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    output_root = Path(args.output_dir)
    for name in ("epochs", "seeds", "models", "predictions"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    setting = f"{experiment.model_name}_s{seed}"
    epoch_path = output_root / "epochs" / f"{setting}.csv"
    model_path = output_root / "models" / f"{setting}.pth"
    result_path = output_root / "seeds" / f"{setting}.json"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.time()

    with epoch_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["epoch", "learning_rate", "train_loss", "train_accuracy",
                      "train_balanced_accuracy", "elapsed_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            lr = args.lr * learning_rate_factor(epoch, args.epochs, args.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
            training = train_epoch(model, train_loader, optimizer, weights, device)
            row = {
                "epoch": epoch, "learning_rate": lr, **training,
                "elapsed_seconds": time.time() - started,
            }
            writer.writerow(row)
            handle.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"[{experiment.experiment_id} s{seed}] ep={epoch:03d} "
                      f"loss={training['train_loss']:.4f} "
                      f"bacc={training['train_balanced_accuracy']:.4f}", flush=True)

    # No validation call occurs before this point.
    validation_metrics, validation_labels, validation_scores = evaluate_once(
        model, validation_loader, device
    )
    train_sha256 = split_digest(Path(args.dataset_root) / "train")
    validation_sha256 = split_digest(Path(args.dataset_root) / "test")
    result = {
        "experiment_id": experiment.experiment_id,
        "model_name": experiment.model_name,
        "description": experiment.description,
        "components": {
            "lm_encoder": experiment.use_lm,
            "gnn_encoder": experiment.use_gnn,
            "condition_lm": experiment.condition_lm,
            "condition_gnn": experiment.condition_gnn,
            "fusion": experiment.fusion,
            "contrastive_learning": False,
            "domain_alignment": False,
            "descriptor_classifier": False,
            "physicochemical_token": experiment.use_physchem,
            "physicochemical_mode": experiment.physchem_mode,
        },
        "seed": seed,
        "architecture_version": "nsanet_v1_physchem_pair_residual",
        "protocol": "NSA_BENCH_UNIFIED_V2",
        "training_protocol": (f"fixed_{args.epochs}_epochs; weighted_CE; "
                              "no_early_stop; final_validation_only"),
        "train_sha256": train_sha256,
        "validation_sha256": validation_sha256,
        "validation_access": "final_evaluation_only",
        "release_status": "EXPLORATORY_PRIVATE_TRAIN",
        "train_class_counts_0_1": counts,
        "class_weights_0_1": weights.detach().cpu().tolist(),
        "descriptor_mean": descriptor_mean.tolist(),
        "descriptor_std": descriptor_std.tolist(),
        "physchem_dimension": (int(len(physchem_mean))
                               if physchem_mean is not None else 0),
        "parameter_count": parameter_count,
        "epochs": args.epochs,
        "validation": validation_metrics,
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                               if device.type == "cuda" else 0.0),
        "epoch_log": str(epoch_path),
    }
    torch.save({"model_state": model.state_dict(), "result": result}, model_path)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    prediction_path = output_root / "predictions" / f"{setting}.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "label", "probability"])
        writer.writerows(zip(range(len(validation_labels)), validation_labels,
                             validation_scores))
    metric = validation_metrics
    print(f"[{experiment.experiment_id} s{seed}] VAL "
          f"AUC={metric['roc_auc']:.4f} AP={metric['average_precision']:.4f} "
          f"BAcc={metric['balanced_accuracy']:.4f} MCC={metric['mcc']:.4f} "
          f"EF10={metric['enrichment_factor_at_10pct']:.2f} "
          f"CondDelta={metric['condition_shuffle_mean_abs_probability_delta']:.5f} "
          f"peak={result['peak_gpu_memory_mb']:.0f}MB", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="+", default=["NSA-Net-C"],
                        help="model name from config.py or all")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="path to a dataset root containing train/ and test/",
    )
    parser.add_argument("--output-dir", default=str(HERE / "results"))
    args = parser.parse_args()
    experiment_ids = list(EXPERIMENTS) if "all" in args.experiment else args.experiment
    unknown = [value for value in experiment_ids if value not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown experiments: {unknown}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"device={device} experiments={experiment_ids} seeds={args.seeds}")
    for experiment_id in experiment_ids:
        for seed in args.seeds:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_seed(EXPERIMENTS[experiment_id], seed, args, device)


if __name__ == "__main__":
    main()
