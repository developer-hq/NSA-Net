"""Freeze a 50-row open benchmark grouped by DOI and unordered molecule pair."""

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dataset_private_train_open_validation"
OPEN_MANIFEST = HERE / "open_validation_97.csv"
PRIVATE_MANIFEST = SOURCE / "private_train_manifest.csv"
PUBLIC_OUTPUT = HERE / "dataset_public_train47_benchmark50"
CLOSED_OUTPUT = HERE / "dataset_closed_train_open_benchmark50"
SELECTION_SEED = 3407
TARGET_ROWS = 50
TARGET_POSITIVES = 20


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_components(rows, smiles_a, smiles_b):
    parent = list(range(len(rows)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    by_doi = defaultdict(list)
    by_pair = defaultdict(list)
    for index, row in enumerate(rows):
        doi = row["doi"].strip().lower()
        if doi:
            by_doi[doi].append(index)
        by_pair[tuple(sorted((smiles_a[index], smiles_b[index])))].append(index)
    for groups in (by_doi, by_pair):
        for indices in groups.values():
            for index in indices[1:]:
                union(indices[0], index)
    components = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    return list(components.values())


def choose_components(rows, components):
    rng = np.random.default_rng(SELECTION_SEED)
    order = rng.permutation(len(components)).tolist()
    states = {(0, 0): []}
    for component_index in order:
        indices = components[component_index]
        size = len(indices)
        positives = sum(int(rows[index]["label"]) for index in indices)
        for (current_size, current_positive), selected in list(states.items())[::-1]:
            key = current_size + size, current_positive + positives
            if key[0] <= TARGET_ROWS and key not in states:
                states[key] = selected + [component_index]
    target = TARGET_ROWS, TARGET_POSITIVES
    if target not in states:
        raise RuntimeError(f"no grouped subset satisfying {target}")
    return sorted(index for component_index in states[target]
                  for index in components[component_index])


def load_arrays(split):
    directory = SOURCE / split
    return {
        path.stem: np.load(path, allow_pickle=True)
        for path in sorted(directory.glob("*.npy"))
    }


def select_arrays(arrays, indices):
    return {name: values[np.asarray(indices)] for name, values in arrays.items()}


def concatenate_arrays(parts):
    names = set(parts[0])
    if any(set(part) != names for part in parts[1:]):
        raise ValueError("array fields differ between source splits")
    return {name: np.concatenate([part[name] for part in parts], axis=0)
            for name in sorted(names)}


def write_split(output, split, arrays, lm_sources):
    directory = output / split
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", values, allow_pickle=values.dtype == object)
    lm = {}
    for source in lm_sources:
        with source.open("rb") as handle:
            lm.update(pickle.load(handle))
    used = {str(value) for key in ("mol_0_smiles", "mol_1_smiles")
            for value in arrays[key]}
    missing = sorted(used - set(lm))
    if missing:
        raise ValueError(f"missing LM embeddings: {missing}")
    with (directory / "mol_LM.pkl").open("wb") as handle:
        pickle.dump({smiles: lm[smiles] for smiles in used}, handle)


def digest(arrays):
    value = hashlib.sha256()
    for name in sorted(arrays):
        value.update(name.encode())
        value.update(np.ascontiguousarray(arrays[name]).tobytes()
                     if arrays[name].dtype != object
                     else repr(arrays[name].tolist()).encode())
    return value.hexdigest()


def main():
    global SOURCE, OPEN_MANIFEST, PRIVATE_MANIFEST, PUBLIC_OUTPUT, CLOSED_OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="dataset root containing train/ and test/")
    parser.add_argument("--open-manifest", required=True,
                        help="CSV manifest for the open validation rows")
    parser.add_argument("--output-root", required=True,
                        help="directory for the generated public and closed splits")
    args = parser.parse_args()
    SOURCE = Path(args.source).resolve()
    OPEN_MANIFEST = Path(args.open_manifest).resolve()
    PRIVATE_MANIFEST = SOURCE / "private_train_manifest.csv"
    output_root = Path(args.output_root).resolve()
    PUBLIC_OUTPUT = output_root / "dataset_public_train47_benchmark50"
    CLOSED_OUTPUT = output_root / "dataset_closed_train_open_benchmark50"
    open_rows = read_csv(OPEN_MANIFEST)
    private_rows = read_csv(PRIVATE_MANIFEST)
    open_arrays = load_arrays("test")
    private_arrays = load_arrays("train")
    open_a = open_arrays["mol_0_smiles"].astype(str)
    open_b = open_arrays["mol_1_smiles"].astype(str)
    private_a = private_arrays["mol_0_smiles"].astype(str)
    private_b = private_arrays["mol_1_smiles"].astype(str)
    if len(open_rows) != len(open_a) or len(private_rows) != len(private_a):
        raise ValueError("manifest and array lengths differ")

    components = union_components(open_rows, open_a, open_b)
    validation_indices = choose_components(open_rows, components)
    validation_set = set(validation_indices)
    public_train_indices = [index for index in range(len(open_rows))
                            if index not in validation_set]
    validation_pairs = {
        tuple(sorted((open_a[index], open_b[index])))
        for index in validation_indices
    }
    validation_dois = {
        open_rows[index]["doi"].strip().lower()
        for index in validation_indices if open_rows[index]["doi"].strip()
    }
    private_keep = []
    private_removed = []
    for index, row in enumerate(private_rows):
        pair = tuple(sorted((private_a[index], private_b[index])))
        doi = row["文献doi"].strip().lower()
        reasons = []
        if pair in validation_pairs:
            reasons.append("benchmark_pair_overlap")
        if doi and doi in validation_dois:
            reasons.append("benchmark_doi_overlap")
        if reasons:
            private_removed.append({**row, "removal_reason": ";".join(reasons)})
        else:
            private_keep.append(index)

    validation = select_arrays(open_arrays, validation_indices)
    public_train = select_arrays(open_arrays, public_train_indices)
    closed_train = concatenate_arrays([
        select_arrays(private_arrays, private_keep), public_train
    ])
    lm_sources = [SOURCE / "train/mol_LM.pkl", SOURCE / "test/mol_LM.pkl"]
    for output, train in ((PUBLIC_OUTPUT, public_train),
                          (CLOSED_OUTPUT, closed_train)):
        write_split(output, "train", train, lm_sources)
        write_split(output, "test", validation, lm_sources)

    validation_manifest = []
    for index in validation_indices:
        validation_manifest.append({
            **open_rows[index], "benchmark_index": index,
            "split": "benchmark50",
            "selection_rule": "doi_and_unordered_pair_grouped_seed3407_50rows_20positive",
        })
    public_train_manifest = []
    for index in public_train_indices:
        public_train_manifest.append({
            **open_rows[index], "benchmark_index": index,
            "split": "public_train47",
            "selection_rule": "complement_of_frozen_benchmark50",
        })
    write_csv(PUBLIC_OUTPUT / "benchmark_manifest.csv", validation_manifest)
    write_csv(PUBLIC_OUTPUT / "train_manifest.csv", public_train_manifest)
    write_csv(CLOSED_OUTPUT / "benchmark_manifest.csv", validation_manifest)
    write_csv(CLOSED_OUTPUT / "public_train_manifest.csv", public_train_manifest)
    if private_removed:
        write_csv(CLOSED_OUTPUT / "removed_private_overlap.csv", private_removed)

    validation_labels = np.asarray(validation["labels"]).reshape(-1).astype(int)
    public_labels = np.asarray(public_train["labels"]).reshape(-1).astype(int)
    closed_labels = np.asarray(closed_train["labels"]).reshape(-1).astype(int)
    if len(validation_labels) != 50 or validation_labels.sum() != 20:
        raise AssertionError("unexpected benchmark composition")
    if not all(row["open"].strip().lower() == "true" for row in validation_manifest):
        raise AssertionError("benchmark contains a non-open row")
    if any(int(row["label"]) == 1 and not row["doi"].strip()
           for row in validation_manifest):
        raise AssertionError("benchmark positive without DOI")

    summary = {
        "selection_seed": SELECTION_SEED,
        "grouping": ["doi", "unordered_smiles_pair"],
        "benchmark": {
            "rows": len(validation_labels), "positive": int(validation_labels.sum()),
            "negative": int((validation_labels == 0).sum()),
            "all_open": True, "all_positive_have_doi": True,
            "sha256": digest(validation),
        },
        "public_train": {
            "rows": len(public_labels), "positive": int(public_labels.sum()),
            "negative": int((public_labels == 0).sum()),
            "all_open": True, "sha256": digest(public_train),
        },
        "closed_train": {
            "rows": len(closed_labels), "positive": int(closed_labels.sum()),
            "negative": int((closed_labels == 0).sum()),
            "private_rows_removed_for_overlap": len(private_removed),
            "sha256": digest(closed_train),
        },
    }
    for output in (PUBLIC_OUTPUT, CLOSED_OUTPUT):
        with (output / "split_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
