"""Mordred descriptor utilities for NSA-Net preprocessing."""

import numpy as np
import torch


def load_mordred_descriptors(smiles):
    """
    Compute Mordred descriptors (1613 features) for a molecule.
    Falls back to a subset of RDKit 2D descriptors if Mordred is not available.
    """
    try:
        from mordred import Calculator, descriptors as mordred_desc
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        calc = Calculator(mordred_desc, ignore_3D=True)
        result = calc(mol)
        desc_vals = []
        for val in result.fill_missing():
            try:
                v = float(val) if val is not None else 0.0
                if not np.isfinite(v):
                    v = 0.0
                desc_vals.append(v)
            except (TypeError, ValueError):
                desc_vals.append(0.0)
        desc = np.array(desc_vals, dtype=np.float32)
        return torch.tensor(desc, dtype=torch.float)

    except ImportError:
        return _compute_rdkit_descriptors(smiles)


def _compute_rdkit_descriptors(smiles, target_dim=1613):
    """Fallback descriptor computation using RDKit when Mordred is unavailable."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descs = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            v = float(val) if val is not None else 0.0
            if not np.isfinite(v):
                v = 0.0
            descs.append(v)
        except Exception:
            descs.append(0.0)

    arr = np.array(descs, dtype=np.float32)
    if len(arr) < target_dim:
        arr = np.pad(arr, (0, target_dim - len(arr)), 'constant')
    else:
        arr = arr[:target_dim]

    return torch.tensor(arr, dtype=torch.float)

