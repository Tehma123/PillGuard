"""Reference vectors ("prototypes") for every seen drug, built from training crops.

VAIPE ships no reference photo per drug, so the reference is the embedding space itself:
for each class we keep the normalised mean of its train-crop embeddings plus up to
``PROTOTYPES_PER_CLASS`` k-means sub-centres (a pill photographed on its blank side and on
its imprinted side do not share one centre). A pill's similarity to a class is the max
cosine over that class's vectors.

Files::

    artifacts/embed/<run>/prototypes.npz     class_ids, vectors, offsets   (numpy)
    web/data/prototypes.json                 {"dim", "classes": [{"id", "name", "vectors"}]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pillguard.config import PROTOTYPES_PER_CLASS


def build_prototypes(embs: np.ndarray, labels: np.ndarray, k: int = PROTOTYPES_PER_CLASS,
                     min_per_centre: int = 8, seed: int = 0) -> dict[int, np.ndarray]:
    """class id -> (k_c, D) array of unit vectors; the first row is always the class mean."""
    from sklearn.cluster import KMeans

    protos: dict[int, np.ndarray] = {}
    for c in sorted(set(int(l) for l in labels)):
        v = embs[labels == c].astype(np.float32)
        mean = v.mean(0)
        rows = [mean / (np.linalg.norm(mean) + 1e-9)]
        kc = min(k - 1, len(v) // min_per_centre)
        if kc >= 2:
            km = KMeans(n_clusters=kc, n_init=4, random_state=seed).fit(v)
            for centre in km.cluster_centers_:
                rows.append(centre / (np.linalg.norm(centre) + 1e-9))
        protos[c] = np.stack(rows).astype(np.float32)
    return protos


def save_prototypes_npz(path: Path, protos: dict[int, np.ndarray]) -> None:
    ids = sorted(protos)
    vectors = np.concatenate([protos[c] for c in ids]) if ids else np.zeros((0, 0), np.float32)
    offsets = np.cumsum([0] + [len(protos[c]) for c in ids])
    np.savez(path, class_ids=np.array(ids, dtype=np.int64), vectors=vectors, offsets=offsets)


def load_prototypes_npz(path: Path) -> dict[int, np.ndarray]:
    z = np.load(path)
    ids, vec, off = z["class_ids"], z["vectors"], z["offsets"]
    return {int(c): vec[off[i]:off[i + 1]] for i, c in enumerate(ids)}


def prototypes_to_json(protos: dict[int, np.ndarray], names: dict[int, str], decimals: int = 4) -> dict:
    ids = sorted(protos)
    dim = int(next(iter(protos.values())).shape[1]) if ids else 0
    return {
        "dim": dim,
        "classes": [{"id": int(c), "name": names.get(c, f"class_{c}"),
                     "vectors": np.round(protos[c], decimals).tolist()} for c in ids],
    }


def prototypes_from_json(d: dict) -> dict[int, np.ndarray]:
    return {int(c["id"]): np.asarray(c["vectors"], dtype=np.float32) for c in d["classes"]}


def save_prototypes_json(path: Path, protos: dict[int, np.ndarray], names: dict[int, str]) -> None:
    Path(path).write_text(json.dumps(prototypes_to_json(protos, names), ensure_ascii=False), encoding="utf-8")
