# -*- coding: utf-8 -*-
"""
benchmark.xlsx → NSA-Net 模型训练数据集

从 benchmark.xlsx 的 final_train / val sheet 生成可直接用于模型训练的
npy/pkl 数据文件，包含图特征、SMILES编码、数值描述符、LM特征和标签。

流水线:
  1. 读取 Excel → 提取 SMILES、描述符、标签
  2. 提取分子图特征 (atom features + adjacency)
  3. 提取 SMILES 字符编码 (words)
  4. 提取预训练语言模型特征 (LM)
  5. 保存为 npy/pkl 文件

用法:
    python benchmark_to_dataset.py
    python benchmark_to_dataset.py --input benchmark.xlsx --output dataset_benchmark --mock-lm
    python benchmark_to_dataset.py --label-col label_nm
"""

import argparse
import math
import os
import sys
import warnings
import re
import pickle

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from nsanet.graph_features import atom_features, N_MOLECULES, DESCRIPTOR_DIM

warnings.filterwarnings('ignore')

# ============================================================
# 常量
# ============================================================
BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}


# ============================================================
# 辅助函数
# ============================================================
def parse_header_normalize(col_name):
    """列名标准化"""
    return re.sub(r'\s+', '', col_name.strip())


def smile_to_graph(smile):
    """SMILES → 原子特征矩阵 + 邻接矩阵"""
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])
    return node_features, adjacency


def smile_to_words(smile):
    """SMILES → 字符编码序列"""
    words = [CHAR_SMI_SET[smile[i]] for i in range(len(smile)) if smile[i] in CHAR_SMI_SET]
    return np.array(words)


def extract_lm_features(smiles_list, tokenizer, model, device, cache_dict=None):
    """批量提取预训练语言模型特征"""
    if cache_dict is None:
        cache_dict = {}

    unique_smiles = [s for s in smiles_list if s not in cache_dict]
    if not unique_smiles:
        return cache_dict

    failed_count = 0
    for i, smiles in enumerate(unique_smiles):
        if (i + 1) % 10 == 0:
            print(f'  LM特征提取: {i+1}/{len(unique_smiles)}')
        try:
            clean_smiles = ''.join(c for c in smiles if c in CHAR_SMI_SET or c in 'CHNOSPBFI')
            if not clean_smiles:
                clean_smiles = smiles
            chem_input = tokenizer(
                [clean_smiles], add_special_tokens=True, padding=True,
                max_length=100, truncation=True, return_tensors='pt'
            )
            c_IDS = chem_input["input_ids"].to(device)
            c_a_m = chem_input["attention_mask"].to(device)
            with torch.no_grad():
                chem_outputs = model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').numpy()
            cache_dict[smiles] = chem_feature
        except Exception as e:
            failed_count += 1
            if failed_count <= 5 or failed_count % 10 == 0:
                print(f"  Warning [{failed_count}]: LM特征提取失败: {smiles[:60]}... | {e}")
            seq_len = min(max(len(smiles), 10), 100)
            cache_dict[smiles] = np.random.randn(seq_len, 768).astype(np.float32) * 0.01

    if failed_count > 0:
        print(f"  LM特征提取完成: {len(unique_smiles)-failed_count}成功, {failed_count}失败(已用随机特征替代)")
    return cache_dict


def extract_mock_lm_features(smiles_list):
    """随机 LM 特征（测试用）"""
    cache_dict = {}
    for smiles in smiles_list:
        seq_len = min(max(len(smiles), 10), 100)
        cache_dict[smiles] = np.random.randn(seq_len, 768).astype(np.float32) * 0.01
    return cache_dict


# ============================================================
# 列名检测
# ============================================================
def find_col(col_map, patterns):
    """从标准化列名映射中查找列"""
    for p in patterns:
        if p in col_map:
            return col_map[p]
    return None


# ============================================================
# 主处理：从 DataFrame 到数据集
# ============================================================
def process_sheet(df, label_col_name, n_molecules):
    """
    从一个 sheet 的 DataFrame 提取数据列表

    Returns:
        data_list: list of dicts, 每个元素:
            smiles: [str] * N_MOLECULES
            descriptors: [float] * DESCRIPTOR_DIM
            label: float (0 or 1)
    """
    # 标准化列名
    col_map = {}
    for c in df.columns:
        col_map[parse_header_normalize(c)] = c

    # 检测 SMILES 列
    smiles_cols = []
    for letter in ['A', 'B', 'C', 'D', 'E']:
        key = None
        for pattern in [f'chem{letter}_SMILES', f'chem_{letter}_SMILES']:
            if pattern in col_map:
                key = col_map[pattern]
                break
        # 备选：用名称列（后续需要 SMILES 映射）
        if key is None:
            for pattern in [f'chem{letter}', f'chem_{letter}']:
                if pattern in col_map:
                    key = col_map[pattern]
                    break
        if key is not None:
            smiles_cols.append(key)
        else:
            break

    if len(smiles_cols) < n_molecules:
        print(f"  Warning: 只找到 {len(smiles_cols)} 个分子列，需要 {n_molecules} 个")

    # 检测比例列
    ratio_cols = []
    for letter in ['A', 'B', 'C', 'D', 'E']:
        key = None
        for pattern in [f'chem{letter}比例', f'chem_{letter}_比例']:
            if pattern in col_map:
                key = col_map[pattern]
                break
        ratio_cols.append(key)

    # 检测条件列
    conc_col = find_col(col_map, [
        '制剂总浓度mM', '制剂总浓度（mM)', '总浓度mM', '浓度mM'
    ])
    ph_col = find_col(col_map, [
        'pH值溶液酸碱度', 'pH值|溶液酸碱度', 'pH值', 'pH', '溶液酸碱度'
    ])
    temp_col = find_col(col_map, [
        '温度(℃)|聚合温度', '温度（℃)|聚合温度', '温度℃聚合温度',
        '温度℃|聚合温度', '温度(℃)', '温度（℃)', '聚合温度'
    ])
    time_col = find_col(col_map, [
        '聚合时间min反应时间', '聚合时间(min)|反应时间',
        '聚合时间(min)', '聚合时间（min)', '聚合时间min', '反应时间'
    ])

    # 检测标签列
    label_col = find_col(col_map, [parse_header_normalize(label_col_name)])
    if label_col is None:
        print(f"  ERROR: 标签列 '{label_col_name}' 未找到！")
        print(f"  可用列: {list(df.columns)}")
        print(f"  标准化列名: {list(col_map.keys())}")
        raise ValueError(f"标签列 '{label_col_name}' 不存在，请用 --label-col 指定正确的列名")

    print(f"  SMILES列: {smiles_cols[:n_molecules]}")
    print(f"  比例列: {[c for c in ratio_cols[:n_molecules] if c]}")
    print(f"  浓度列: {conc_col}")
    print(f"  pH列: {ph_col}")
    print(f"  温度列: {temp_col}")
    print(f"  时间列: {time_col}")
    print(f"  标签列: {label_col}")

    # 逐行处理
    data_list = []
    skipped = 0
    for idx, row in df.iterrows():
        # SMILES
        smiles = []
        valid = True
        for m_idx in range(n_molecules):
            val = str(row.get(smiles_cols[m_idx], '')).strip()
            if val in ('nan', 'None', '', 'NaN'):
                valid = False
                break
            smiles.append(val)
        if not valid:
            skipped += 1
            continue

        # 比例 → 归一化为相对于分子0的比值
        ratios = []
        base_ratio = None
        for m_idx in range(n_molecules):
            col = ratio_cols[m_idx]
            if col is None:
                ratios.append(1.0)
                continue
            r = str(row.get(col, '')).strip()
            try:
                ratios.append(float(r))
            except (ValueError, TypeError):
                ratios.append(1.0)

        # 归一化
        base = ratios[0] if ratios[0] != 0 else 1.0
        ratios = [r / base for r in ratios]

        # 条件描述符
        def safe_float(col, default=0.0):
            if col is None:
                return default
            v = str(row.get(col, '')).strip()
            if v in ('nan', 'None', '', 'NaN'):
                return default
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        conc = safe_float(conc_col)
        ph = safe_float(ph_col)
        temp = safe_float(temp_col)
        time_val = safe_float(time_col)

        descriptors = ratios + [conc, ph, temp, time_val]

        # 检查描述符维度
        if len(descriptors) != DESCRIPTOR_DIM:
            print(f"  Warning: Row {idx} descriptor dim={len(descriptors)}, expected={DESCRIPTOR_DIM}")
            skipped += 1
            continue

        # 过滤 NaN/Inf
        if any(math.isnan(v) or math.isinf(v) for v in descriptors):
            skipped += 1
            continue

        # 标签
        label = None
        if label_col:
            lv = str(row.get(label_col, '')).strip()
            try:
                label = int(float(lv))
            except (ValueError, TypeError):
                skipped += 1
                continue

        if label is None:
            skipped += 1
            continue

        data_list.append({
            'smiles': smiles,
            'descriptors': descriptors,
            'label': label,
        })

    print(f"  有效样本: {len(data_list)}, 跳过: {skipped}")
    return data_list


def build_dataset(data_list, output_dir, use_mock_lm=False):
    """
    从 data_list 构建 npy/pkl 数据集文件

    输出文件结构:
        {output_dir}/
            train/  或  test/
                mol_0_words.npy
                mol_0_atoms.npy
                mol_0_adjs.npy
                mol_0_smiles.npy
                mol_1_words.npy
                ...
                descriptors.npy
                labels.npy
                mol_LM.pkl
    """
    N = len(data_list)
    print(f"\n构建数据集: {N} 样本, {N_MOLECULES} 分子, 描述符维度={DESCRIPTOR_DIM}")

    # 初始化
    mol_atoms = [[] for _ in range(N_MOLECULES)]
    mol_adjs = [[] for _ in range(N_MOLECULES)]
    mol_words = [[] for _ in range(N_MOLECULES)]
    mol_smiles_list = [[] for _ in range(N_MOLECULES)]
    descriptors = []
    labels = []

    # 提取图特征和字符编码
    for no, item in enumerate(data_list):
        if (no + 1) % 50 == 0:
            print(f'  图特征提取: {no+1}/{N}')

        for m_idx in range(N_MOLECULES):
            smiles = item['smiles'][m_idx]
            mol_smiles_list[m_idx].append(smiles)

            try:
                atom_feat, adj = smile_to_graph(smiles)
                mol_atoms[m_idx].append(atom_feat)
                mol_adjs[m_idx].append(adj)
            except Exception as e:
                print(f"  Warning: Invalid SMILES for mol_{m_idx}: {smiles[:50]} | {e}")
                # 用空特征占位
                mol_atoms[m_idx].append(np.zeros((1, 75), dtype=np.float32))
                mol_adjs[m_idx].append(np.zeros((1, 1), dtype=np.float32))

            mol_words[m_idx].append(smile_to_words(smiles))

        descriptors.append(item['descriptors'])
        labels.append(np.array([item['label']]))

    # 检查数据完整性
    actual_samples = min(len(mol_atoms[0]), len(mol_words[0]), len(descriptors))
    for m_idx in range(N_MOLECULES):
        mol_atoms[m_idx] = mol_atoms[m_idx][:actual_samples]
        mol_adjs[m_idx] = mol_adjs[m_idx][:actual_samples]
        mol_words[m_idx] = mol_words[m_idx][:actual_samples]
        mol_smiles_list[m_idx] = mol_smiles_list[m_idx][:actual_samples]
    descriptors = descriptors[:actual_samples]
    labels = labels[:actual_samples]

    print(f"  有效特征提取: {actual_samples} 样本")

    # 提取 LM 特征
    use_mock = use_mock_lm
    if not use_mock:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('  Using GPU for LM feature extraction...')
        else:
            device = torch.device('cpu')
            print('  Using CPU for LM feature extraction...')

        try:
            from transformers import AutoModel, AutoTokenizer
            print("  Loading pre-trained LM model...")
            chem_tokenizer = AutoTokenizer.from_pretrained(
                "seyonec/PubChem10M_SMILES_BPE_450k", do_lower_case=False)
            chem_model = AutoModel.from_pretrained(
                "seyonec/PubChem10M_SMILES_BPE_450k").to(device)

            all_unique_smiles = set()
            for m_idx in range(N_MOLECULES):
                all_unique_smiles.update(mol_smiles_list[m_idx])
            all_unique_smiles = list(all_unique_smiles)
            print(f"  唯一 SMILES 数量: {len(all_unique_smiles)}")

            mol_LM = extract_lm_features(all_unique_smiles, chem_tokenizer, chem_model, device)
            print("  LM features extracted successfully!")
        except Exception as e:
            print(f"\n  Failed to load LM model: {e}")
            print("  Falling back to mock LM features...")
            use_mock = True

    if use_mock:
        print("  Using mock (random) LM features...")
        all_unique_smiles = set()
        for m_idx in range(N_MOLECULES):
            all_unique_smiles.update(mol_smiles_list[m_idx])
        all_unique_smiles = list(all_unique_smiles)
        mol_LM = extract_mock_lm_features(all_unique_smiles)

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "mol_LM.pkl"), "wb") as f:
        pickle.dump(mol_LM, f)

    for m_idx in range(N_MOLECULES):
        np.save(os.path.join(output_dir, f'mol_{m_idx}_words'),
                np.array(mol_words[m_idx], dtype=object), allow_pickle=True)
        np.save(os.path.join(output_dir, f'mol_{m_idx}_atoms'),
                np.array(mol_atoms[m_idx], dtype=object), allow_pickle=True)
        np.save(os.path.join(output_dir, f'mol_{m_idx}_adjs'),
                np.array(mol_adjs[m_idx], dtype=object), allow_pickle=True)
        np.save(os.path.join(output_dir, f'mol_{m_idx}_smiles'),
                np.array(mol_smiles_list[m_idx], dtype=object), allow_pickle=True)

    np.save(os.path.join(output_dir, 'descriptors'),
            np.array(descriptors, dtype=object), allow_pickle=True)
    np.save(os.path.join(output_dir, 'labels'),
            np.array(labels, dtype=object), allow_pickle=True)

    # 统计
    label_arr = np.array([l[0] for l in labels])
    pos = (label_arr == 1).sum()
    neg = (label_arr == 0).sum()
    print(f"\n  保存至: {output_dir}")
    for m_idx in range(N_MOLECULES):
        print(f"    mol_{m_idx}: {len(mol_words[m_idx])} samples")
    print(f"    descriptors: {len(descriptors)}")
    print(f"    labels: {len(labels)} (1={pos}, 0={neg})")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='从 benchmark.xlsx 生成 NSA-Net 模型训练数据集')
    parser.add_argument('--input', '-i', default='benchmark.xlsx',
                        help='输入 Excel 文件 (default: benchmark.xlsx)')
    parser.add_argument('--output', '-o', default='dataset_benchmark',
                        help='输出目录 (default: dataset_benchmark)')
    parser.add_argument('--train-sheet', default='final_train',
                        help='训练集 sheet 名 (default: final_train)')
    parser.add_argument('--val-sheet', default='val',
                        help='验证集 sheet 名 (default: val)')
    parser.add_argument('--label-col', '-l', default='label',
                        help='标签列名 (default: label)')
    parser.add_argument('--n-molecules', '-n', type=int, default=2,
                        help='分子数量 (default: 2)')
    parser.add_argument('--mock-lm', action='store_true',
                        help='使用随机 LM 特征（无需联网下载模型）')

    args = parser.parse_args()

    # Resolve relative paths from the current working directory.
    input_path = os.path.abspath(args.input)

    if not os.path.exists(input_path):
        print(f"Error: 文件不存在: {input_path}")
        sys.exit(1)

    print(f"读取: {input_path}")
    xls = pd.ExcelFile(input_path)

    # 修改 graph_features 中的全局变量以匹配分子数
    import nsanet.graph_features as gf
    gf.N_MOLECULES = args.n_molecules
    gf.DESCRIPTOR_DIM = args.n_molecules + 4
    # 重新导入更新后的值
    N_MOLECULES = gf.N_MOLECULES
    DESCRIPTOR_DIM = gf.DESCRIPTOR_DIM

    print(f"配置: N_MOLECULES={N_MOLECULES}, DESCRIPTOR_DIM={DESCRIPTOR_DIM}")
    print(f"标签列: {args.label_col}")
    print(f"训练集 sheet: {args.train_sheet}")
    print(f"验证集 sheet: {args.val_sheet}")

    # 处理训练集
    print(f"\n{'='*50}")
    print(f"处理训练集: {args.train_sheet}")
    print(f"{'='*50}")
    df_train = pd.read_excel(xls, sheet_name=args.train_sheet)
    train_data = process_sheet(df_train, args.label_col, N_MOLECULES)

    # 处理验证集
    print(f"\n{'='*50}")
    print(f"处理验证集: {args.val_sheet}")
    print(f"{'='*50}")
    df_val = pd.read_excel(xls, sheet_name=args.val_sheet)
    val_data = process_sheet(df_val, args.label_col, N_MOLECULES)

    # Build the output dataset outside the source tree unless explicitly requested.
    output_base = os.path.abspath(args.output)

    print(f"\n{'='*50}")
    print(f"构建训练集")
    print(f"{'='*50}")
    build_dataset(train_data, os.path.join(output_base, 'train'), args.mock_lm)

    print(f"\n{'='*50}")
    print(f"构建验证集")
    print(f"{'='*50}")
    build_dataset(val_data, os.path.join(output_base, 'test'), args.mock_lm)

    print(f"\n{'='*50}")
    print(f"全部完成!")
    print(f"  训练集: {os.path.join(output_base, 'train')} ({len(train_data)} 样本)")
    print(f"  验证集: {os.path.join(output_base, 'test')} ({len(val_data)} 样本)")
    print(f"  N_MOLECULES={N_MOLECULES}, DESCRIPTOR_DIM={DESCRIPTOR_DIM}")
    print(f"  使用方法: 修改 src/nsanet/graph_features.py 中 N_MOLECULES={N_MOLECULES}")
    print(f"{'='*50}")
