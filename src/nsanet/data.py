import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NSADataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = Path(split_dir)
        self.words = [np.load(self.split_dir / f"mol_{i}_words.npy", allow_pickle=True)
                      for i in range(2)]
        self.atoms = [np.load(self.split_dir / f"mol_{i}_atoms.npy", allow_pickle=True)
                      for i in range(2)]
        self.adjs = [np.load(self.split_dir / f"mol_{i}_adjs.npy", allow_pickle=True)
                    for i in range(2)]
        self.smiles = [np.load(self.split_dir / f"mol_{i}_smiles.npy", allow_pickle=True)
                       for i in range(2)]
        self.descriptors = np.load(self.split_dir / "descriptors.npy", allow_pickle=True)
        self.physchem = [
            np.load(self.split_dir / f"mol_{i}_mordred.npy", allow_pickle=False)
            if (self.split_dir / f"mol_{i}_mordred.npy").exists() else None
            for i in range(2)
        ]
        self.labels = np.load(self.split_dir / "labels.npy", allow_pickle=True)
        with (self.split_dir / "mol_LM.pkl").open("rb") as handle:
            self.lm = pickle.load(handle)
        lengths = {len(self.labels), len(self.descriptors)}
        lengths.update(len(values) for group in (self.words, self.atoms, self.adjs, self.smiles)
                       for values in group)
        if len(lengths) != 1:
            raise ValueError(f"inconsistent split lengths: {lengths}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "atoms": [self.atoms[m][index] for m in range(2)],
            "adjs": [self.adjs[m][index] for m in range(2)],
            "smiles": [str(self.smiles[m][index]) for m in range(2)],
            "descriptor": np.asarray(self.descriptors[index], dtype=np.float32),
            "physchem": [
                np.asarray(self.physchem[m][index], dtype=np.float32)
                if self.physchem[m] is not None else None for m in range(2)
            ],
            "label": int(np.asarray(self.labels[index]).reshape(-1)[0]),
        }


def descriptor_statistics(dataset):
    descriptors = np.stack([np.asarray(value, dtype=np.float32)
                            for value in dataset.descriptors])
    mean = descriptors.mean(axis=0)
    std = descriptors.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def physchem_statistics(dataset):
    if any(values is None for values in dataset.physchem):
        return None, None
    values = np.concatenate(dataset.physchem, axis=0).astype(np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class BatchCollator:
    def __init__(self, lm_dict, descriptor_mean, descriptor_std,
                 physchem_mean=None, physchem_std=None, max_lm_tokens=100):
        self.lm_dict = lm_dict
        self.descriptor_mean = torch.as_tensor(descriptor_mean, dtype=torch.float32)
        self.descriptor_std = torch.as_tensor(descriptor_std, dtype=torch.float32)
        self.physchem_mean = (torch.as_tensor(physchem_mean, dtype=torch.float32)
                              if physchem_mean is not None else None)
        self.physchem_std = (torch.as_tensor(physchem_std, dtype=torch.float32)
                             if physchem_std is not None else None)
        self.max_lm_tokens = max_lm_tokens

    @staticmethod
    def _graphs(samples, molecule_index):
        lengths = [int(np.asarray(sample["atoms"][molecule_index]).shape[0]) for sample in samples]
        max_atoms = max(lengths)
        atoms = torch.zeros(len(samples), max_atoms, 75, dtype=torch.float32)
        adjacency = torch.zeros(len(samples), max_atoms, max_atoms, dtype=torch.float32)
        mask = torch.zeros(len(samples), max_atoms, dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            atom_values = torch.as_tensor(np.asarray(sample["atoms"][molecule_index], dtype=np.float32))
            adj_values = torch.as_tensor(np.asarray(sample["adjs"][molecule_index], dtype=np.float32))
            length = lengths[batch_index]
            atoms[batch_index, :length] = atom_values
            adjacency[batch_index, :length, :length] = adj_values
            mask[batch_index, :length] = True
        return {"atoms": atoms, "adjacency": adjacency, "mask": mask}

    def _lm_tokens(self, samples, molecule_index):
        embeddings = []
        lengths = []
        for sample in samples:
            value = self.lm_dict.get(sample["smiles"][molecule_index])
            if value is None:
                value = np.zeros((1, 768), dtype=np.float32)
            value = np.asarray(value, dtype=np.float32)[:self.max_lm_tokens]
            embeddings.append(value)
            lengths.append(max(1, len(value)))
        max_length = max(lengths)
        tokens = torch.zeros(len(samples), max_length, 768, dtype=torch.float32)
        mask = torch.zeros(len(samples), max_length, dtype=torch.bool)
        for batch_index, value in enumerate(embeddings):
            length = len(value)
            tokens[batch_index, :length] = torch.as_tensor(value)
            mask[batch_index, :length] = True
        return {"tokens": tokens, "mask": mask}

    def __call__(self, samples):
        descriptors = torch.stack([
            torch.as_tensor(sample["descriptor"], dtype=torch.float32) for sample in samples
        ])
        descriptors = ((descriptors - self.descriptor_mean) / self.descriptor_std).clamp(-5.0, 5.0)
        physchem = None
        if self.physchem_mean is not None:
            physchem = []
            for molecule_index in range(2):
                values = torch.stack([
                    torch.as_tensor(sample["physchem"][molecule_index], dtype=torch.float32)
                    for sample in samples
                ])
                values = ((values - self.physchem_mean) / self.physchem_std).clamp(-5.0, 5.0)
                physchem.append(values)
        return {
            "graphs": [self._graphs(samples, molecule_index) for molecule_index in range(2)],
            "lm": [self._lm_tokens(samples, molecule_index) for molecule_index in range(2)],
            "condition": descriptors,
            "physchem": physchem,
            "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
            "smiles": [[sample["smiles"][molecule_index] for sample in samples]
                       for molecule_index in range(2)],
        }


def create_dataloaders(dataset_root, batch_size, seed):
    root = Path(dataset_root)
    train_dataset = NSADataset(root / "train")
    validation_dataset = NSADataset(root / "test")
    mean, std = descriptor_statistics(train_dataset)
    physchem_mean, physchem_std = physchem_statistics(train_dataset)
    train_collator = BatchCollator(
        train_dataset.lm, mean, std, physchem_mean, physchem_std
    )
    validation_lm = dict(train_dataset.lm)
    validation_lm.update(validation_dataset.lm)
    validation_collator = BatchCollator(
        validation_lm, mean, std, physchem_mean, physchem_std
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=train_collator, generator=generator, drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=validation_collator, drop_last=False,
    )
    return train_loader, validation_loader, mean, std, physchem_mean, physchem_std


def move_batch(batch, device):
    return {
        "graphs": [{key: value.to(device) for key, value in graph.items()}
                   for graph in batch["graphs"]],
        "lm": [{key: value.to(device) for key, value in lm.items()}
               for lm in batch["lm"]],
        "condition": batch["condition"].to(device),
        "physchem": ([value.to(device) for value in batch["physchem"]]
                      if batch["physchem"] is not None else None),
        "labels": batch["labels"].to(device),
        "smiles": batch["smiles"],
    }
