# -*- coding: utf-8 -*-
"""
NSANet 预测脚本

支持两种模式:
  1. 批量模式: 从预处理好的 .npy 数据集预测
  2. 单条模式: 从 SMILES + 描述符实时构建输入预测 (NSANetPredictor)

用法 (批量):
    python scripts/predict.py --checkpoint models/NSA-Net-C.pth --dataset-root /path/to/dataset/test

用法 (单条 / 供 optimize_predict.py 调用):
    predictor = NSANetPredictor(checkpoint_path, dataset_root, ...)
    result = predictor.predict(smiles_list, descriptors)
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nsanet.graph_features import atom_features, N_MOLECULES, DESCRIPTOR_DIM

from rdkit import Chem

from nsanet.config import EXPERIMENTS
from nsanet.data import (NSADataset, BatchCollator, descriptor_statistics,
                  physchem_statistics, move_batch)
from nsanet.metrics import compute_metrics
from nsanet.model import NSANet
def resolve_model_name(checkpoint_path, result_meta):
    checkpoint_stem = Path(checkpoint_path).stem
    candidates = [
        checkpoint_stem,
        result_meta.get("model_name"),
        result_meta.get("experiment_id"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        for model_name in EXPERIMENTS:
            if candidate == model_name or candidate.startswith(f"{model_name}_s"):
                return model_name
    raise ValueError(
        f"cannot resolve model name from checkpoint '{checkpoint_path}'"
    )



# ============================================================
# SMILES → 图特征转换 (与 data_merge.py / train.py 一致)
# ============================================================

def smile_to_graph(smile):
    """从 SMILES 提取分子图特征 (atoms: N×75, adjacency: N×N)"""
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    n_atoms = molecule.GetNumAtoms()
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in molecule.GetAtoms()],
                             dtype=np.float32)
    return node_features, adjacency


def load_lm_pickle(dataset_root):
    """加载训练集的 LM pickle 字典 {SMILES: embedding_array}"""
    lm_path = Path(dataset_root) / "train" / "mol_LM.pkl"
    if not lm_path.exists():
        # 尝试直接在 dataset_root 下查找
        lm_path = Path(dataset_root) / "mol_LM.pkl"
    if not lm_path.exists():
        return None
    with lm_path.open("rb") as f:
        return pickle.load(f)


def get_mock_lm(smiles, max_len=100):
    """生成 mock LM 特征 (与 NanoPolyPredictor._extract_mock_lm 一致)"""
    seq_len = min(max(len(smiles), 10), max_len)
    return np.random.randn(seq_len, 768).astype(np.float32) * 0.01


# ============================================================
# 单条预测器: 从 SMILES + 描述符实时构建 batch
# ============================================================

class NSANetPredictor:
    """
    NSANet 单条/批量预测器

    从 SMILES + 描述符实时构建模型输入，无需预处理 .npy 文件。
    供 optimize_predict.py 等需要动态修改描述符的场景使用。

    用法:
        predictor = NSANetPredictor(
            checkpoint_path='models/NSA-Net-C.pth',
            dataset_root='/path/to/dataset',
        )
        result = predictor.predict(
            smiles_list=["CCO", "c1ccccc1O"],
            descriptors=[1.0, 0.5, 10.0, 7.4, 25.0, 30.0],
        )
        print(f"合格概率: {result['probability']:.2%}")
    """

    def __init__(self, checkpoint_path, dataset_root=None, device=None,
                 d_model=128, n_heads=4, gnn_layers=3, dropout=0.2,
                 use_mock_lm=False):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        result_meta = checkpoint.get("result", {})

        # 提取实验配置
        model_name = resolve_model_name(checkpoint_path, result_meta)
        self.experiment = EXPERIMENTS[model_name]
        self.experiment_id = model_name
        self.seed = result_meta.get("seed")

        # 描述符归一化参数 (从训练时保存)
        self.descriptor_mean = np.asarray(
            result_meta["descriptor_mean"], dtype=np.float32
        )
        self.descriptor_std = np.asarray(
            result_meta["descriptor_std"], dtype=np.float32
        )

        # physchem 归一化参数
        self.physchem_mean = None
        self.physchem_std = None
        physchem_dim = result_meta.get("physchem_dimension", 0)
        if physchem_dim > 0 and dataset_root:
            train_root = Path(dataset_root) / "train"
            if train_root.exists():
                train_dataset = NSADataset(train_root)
                from nsanet.data import physchem_statistics
                self.physchem_mean, self.physchem_std = physchem_statistics(train_dataset)

        # 构建模型
        self.model = NSANet(
            self.experiment, d_model=d_model, n_heads=n_heads,
            gnn_layers=gnn_layers, dropout=dropout,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # 加载 LM 字典
        self.use_mock_lm = use_mock_lm
        self.lm_dict = None
        if not use_mock_lm and dataset_root:
            self.lm_dict = load_lm_pickle(dataset_root)
            if self.lm_dict is None:
                print("WARNING: mol_LM.pkl not found, using mock LM features")
                self.use_mock_lm = True
        elif not use_mock_lm:
            self.use_mock_lm = True

        print(f"NSANetPredictor initialized: "
              f"model={model_name} device={self.device} "
              f"mock_lm={self.use_mock_lm}")

    def _build_graph_batch(self, smiles_list):
        """从 SMILES 列表构建图特征 batch (batch_size=1)"""
        atoms_list = []
        adjs_list = []
        masks = []

        for smile in smiles_list:
            node_features, adjacency = smile_to_graph(smile)
            atoms_list.append(node_features)
            adjs_list.append(adjacency)

        # padding 到最大原子数
        max_atoms = max(a.shape[0] for a in atoms_list)

        batch_atoms = torch.zeros(1, max_atoms, 75, dtype=torch.float32)
        batch_adjs = torch.zeros(1, max_atoms, max_atoms, dtype=torch.float32)
        batch_mask = torch.zeros(1, max_atoms, dtype=torch.bool)

        for mol_idx in range(len(smiles_list)):
            n_atoms = atoms_list[mol_idx].shape[0]
            # 注意: nsanet.data 中每个分子独立 padding
            # 但这里 batch=1, 所以直接填入
            batch_atoms[0, :n_atoms, :] = torch.as_tensor(atoms_list[mol_idx])
            batch_adjs[0, :n_atoms, :n_atoms] = torch.as_tensor(adjs_list[mol_idx])
            batch_mask[0, :n_atoms] = True

        return {"atoms": batch_atoms, "adjacency": batch_adjs, "mask": batch_mask}

    def _build_lm_batch(self, smiles_list):
        """从 SMILES 列表构建 LM 特征 batch (batch_size=1)"""
        max_lm_len = 100
        batch_tokens = torch.zeros(1, max_lm_len, 768, dtype=torch.float32)
        batch_mask = torch.zeros(1, max_lm_len, dtype=torch.bool)

        for mol_idx, smile in enumerate(smiles_list):
            if self.use_mock_lm or self.lm_dict is None:
                emb = get_mock_lm(smile, max_lm_len)
            else:
                emb = self.lm_dict.get(smile)
                if emb is None:
                    emb = get_mock_lm(smile, max_lm_len)
                else:
                    emb = np.asarray(emb, dtype=np.float32)[:max_lm_len]

            length = min(emb.shape[0], max_lm_len)
            batch_tokens[0, :length, :] = torch.as_tensor(emb[:length])
            batch_mask[0, :length] = True

        return {"tokens": batch_tokens, "mask": batch_mask}

    def _build_condition(self, descriptors):
        """构建条件描述符 (归一化)"""
        desc = np.asarray(descriptors, dtype=np.float32)
        if self.descriptor_mean is not None and len(desc) == len(self.descriptor_mean):
            desc = (desc - self.descriptor_mean) / self.descriptor_std
            desc = np.clip(desc, -5.0, 5.0)
        return torch.as_tensor(desc, dtype=torch.float32).unsqueeze(0)  # (1, dim)

    def _build_physchem(self, smiles_list):
        """构建 physchem 特征 (如果实验需要)"""
        if not self.experiment.use_physchem:
            return None
        if self.physchem_mean is None:
            return None
        # 注意: 优化场景下无法实时计算 Mordred 描述符
        # 使用零向量 (模型训练时已做归一化, 零向量 ≈ 均值)
        dim = len(self.physchem_mean)
        values = torch.zeros(1, dim, dtype=torch.float32)
        # 归一化: (0 - mean) / std ≈ -mean/std
        values = ((values - torch.as_tensor(self.physchem_mean)) /
                  torch.as_tensor(self.physchem_std)).clamp(-5.0, 5.0)
        # 返回两个分子的 physchem (相同零值)
        return [values, values.clone()]

    def predict(self, smiles_list, descriptors):
        """
        预测单个分子组合的粒径合格概率

        Args:
            smiles_list: 2个分子的 SMILES 字符串列表
            descriptors: 数值描述符列表 [molar_ratio_0, molar_ratio_1,
                         concentration_mM, pH, temperature_C, incubation_time_min]

        Returns:
            dict: {
                'prediction': 0或1,
                'probability': 合格概率 (0~1),
                'confidence': 预测置信度,
                'smiles_list': 输入的 SMILES 列表,
                'descriptors': 描述符列表
            }
        """
        if len(smiles_list) != N_MOLECULES:
            raise ValueError(f"Expected {N_MOLECULES} SMILES, got {len(smiles_list)}")

        # 构建各模态输入
        graphs = [self._build_graph_batch([smiles_list[i]]) for i in range(N_MOLECULES)]
        lm = [self._build_lm_batch([smiles_list[i]]) for i in range(N_MOLECULES)]
        condition = self._build_condition(descriptors)
        physchem = self._build_physchem(smiles_list)
        labels = torch.tensor([0], dtype=torch.long)  # 占位, 推理时不使用

        batch = {
            "graphs": [{k: v.to(self.device) for k, v in g.items()} for g in graphs],
            "lm": [{k: v.to(self.device) for k, v in l.items()} for l in lm],
            "condition": condition.to(self.device),
            "physchem": ([v.to(self.device) for v in physchem]
                         if physchem is not None else None),
            "labels": labels.to(self.device),
            "smiles": [[s] for s in smiles_list],
        }

        with torch.no_grad():
            output = self.model(batch)
            logits = output["logits"]
            prob = torch.softmax(logits.float(), dim=1)[0, 1].item()

        pred = 1 if prob > 0.5 else 0
        return {
            'prediction': pred,
            'probability': prob,
            'confidence': max(prob, 1 - prob),
            'smiles_list': smiles_list,
            'descriptors': list(descriptors),
        }


# ============================================================
# 批量预测: 从 .npy 数据集
# ============================================================

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def build_model(experiment, checkpoint, device, d_model=128, n_heads=4,
                gnn_layers=3, dropout=0.2):
    model = NSANet(
        experiment, d_model=d_model, n_heads=n_heads,
        gnn_layers=gnn_layers, dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def create_prediction_loader(dataset_root, batch_size, descriptor_mean,
                             descriptor_std, physchem_mean=None,
                             physchem_std=None):
    dataset = NSADataset(Path(dataset_root))
    collator = BatchCollator(
        dataset.lm, descriptor_mean, descriptor_std,
        physchem_mean, physchem_std,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collator, drop_last=False,
    )
    return dataset, loader


@torch.no_grad()
def predict_batch(model, loader, device):
    all_labels = []
    all_scores = []
    all_logits = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(batch)
        logits = output["logits"]
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        all_labels.extend(batch["labels"].cpu().tolist())
        all_scores.extend(probabilities.cpu().tolist())
        all_logits.append(logits.cpu().numpy())
    labels = np.asarray(all_labels, dtype=int)
    scores = np.asarray(all_scores, dtype=float)
    logits = np.concatenate(all_logits, axis=0)
    return labels, scores, logits


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained NSANet model")
    parser.add_argument("--checkpoint", required=True,
                        help="path to .pth checkpoint file")
    parser.add_argument("--dataset-root", required=True,
                        help="path to dataset split directory")
    parser.add_argument("--output-dir", default=None,
                        help="directory for prediction outputs (default: same as checkpoint)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Load checkpoint and extract training metadata
    checkpoint = load_checkpoint(args.checkpoint, device)
    result_meta = checkpoint.get("result", {})
    model_name = resolve_model_name(args.checkpoint, result_meta)
    seed = result_meta.get("seed")
    descriptor_mean = np.asarray(result_meta["descriptor_mean"], dtype=np.float32)
    descriptor_std = np.asarray(result_meta["descriptor_std"], dtype=np.float32)
    physchem_dim = result_meta.get("physchem_dimension", 0)

    # Reconstruct physchem statistics from training dataset if needed
    physchem_mean = None
    physchem_std = None
    if physchem_dim > 0:
        train_root = str(Path(args.dataset_root).parent / "train")
        if Path(train_root).exists():
            train_dataset = NSADataset(train_root)
            physchem_mean, physchem_std = physchem_statistics(train_dataset)
        else:
            print(f"WARNING: training split not found: {train_root}")

    # Resolve experiment config
    experiment = EXPERIMENTS[model_name]
    model = build_model(experiment, checkpoint, device, args.d_model,
                        args.n_heads, args.gnn_layers, args.dropout)
    print(f"Loaded model: model={model_name} seed={seed} device={device}")

    # Create data loader
    dataset = NSADataset(Path(args.dataset_root))
    collator = BatchCollator(
        dataset.lm, descriptor_mean, descriptor_std,
        physchem_mean, physchem_std,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collator, drop_last=False,
    )
    print(f"Dataset: {len(dataset)} samples from {args.dataset_root}")

    # Run prediction
    labels, scores, logits = predict_batch(model, loader, device)

    # Compute metrics if labels are available
    has_labels = not np.all(labels == 0) or np.any(labels == 1)
    metrics = compute_metrics(labels, scores, threshold=0.5) if has_labels else None

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    setting = experiment.model_name
    if seed is not None:
        setting += f"_s{seed}"

    # Save predictions CSV
    prediction_path = output_dir / f"predictions_{setting}.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "label", "probability", "predicted_class"])
        for index in range(len(labels)):
            pred_class = int(scores[index] >= 0.5)
            writer.writerow([index, int(labels[index]), f"{scores[index]:.6f}", pred_class])
    print(f"Predictions saved to {prediction_path}")

    # Save logits
    logits_path = output_dir / f"logits_{setting}.npy"
    np.save(logits_path, logits)
    print(f"Logits saved to {logits_path}")

    # Save metrics JSON
    if metrics:
        metrics_path = output_dir / f"metrics_{setting}.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"Metrics: AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f} "
              f"BAcc={metrics['balanced_accuracy']:.4f} MCC={metrics['mcc']:.4f} "
              f"EF10={metrics['enrichment_factor_at_10pct']:.2f}")
        print(f"Metrics saved to {metrics_path}")

    # Print summary
    n_positive = int(labels.sum())
    n_negative = int((labels == 0).sum())
    print(f"\nSummary: {len(labels)} samples "
          f"(positive={n_positive}, negative={n_negative})")
    print(f"Mean probability: {scores.mean():.4f} "
          f"Median: {np.median(scores):.4f}")


if __name__ == "__main__":
    main()
