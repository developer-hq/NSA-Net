"""Precompute per-molecule Mordred arrays for the clean NSA-Net pipeline."""

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from nsanet.mordred_features import load_mordred_descriptors  # noqa: E402


def compute_split(split_dir):
    split_dir = Path(split_dir)
    smiles_groups = [
        np.load(split_dir / f"mol_{index}_smiles.npy", allow_pickle=True).astype(str)
        for index in range(2)
    ]
    cache = {}
    for smiles in sorted(set(np.concatenate(smiles_groups))):
        values = load_mordred_descriptors(smiles)
        if values is None:
            raise ValueError(f"Mordred failed for {smiles}")
        cache[smiles] = values.numpy().astype(np.float32)
    dimensions = {len(values) for values in cache.values()}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent Mordred dimensions: {dimensions}")
    for molecule_index, smiles_values in enumerate(smiles_groups):
        matrix = np.stack([cache[smiles] for smiles in smiles_values])
        output_path = split_dir / f"mol_{molecule_index}_mordred.npy"
        np.save(output_path, matrix, allow_pickle=False)
        print(f"wrote {output_path}: {matrix.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="path to a dataset root containing train/ and test/",
    )
    args = parser.parse_args()
    root = Path(args.dataset_root)
    compute_split(root / "train")
    compute_split(root / "test")


if __name__ == "__main__":
    main()
