"""Assemble everything the browser demo loads under ``web/models`` and ``web/data``.

    web/models/detector.int8.onnx, embed.int8.onnx
    web/data/config.json        shared constants + decision parameters + model file names
    web/data/prototypes.json    reference vectors per drug
    web/data/drugs.json         [{id, name}] for the drug picker (seen drugs only)
    web/data/samples.json       a few prescriptions from the test split as ready-made drug lists
                                (drug ids and names only; no dataset images are shipped)

Usage::

    python -m pillguard.web.export_assets --detector artifacts/export/detector.int8.onnx \
        --embed artifacts/export/embed.int8.onnx --prototypes artifacts/decision/prototypes.npz \
        --params artifacts/decision/params.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from pillguard import __version__
from pillguard.config import WEB_DIR, DecisionParams, SharedConfig
from pillguard.data.splits import Split
from pillguard.data.vaipe import load_aliases, load_classes, load_pill_pres_map, prescription_drug_table
from pillguard.embed.prototypes import load_prototypes_npz, save_prototypes_json


def sample_prescriptions(root: Path, split: Split, seen: set[int], names: dict[int, str], n: int = 24) -> list[dict]:
    pres_drugs = prescription_drug_table(root)
    p2p = {}
    for pres, pills in load_pill_pres_map(root).items():
        for f in pills:
            p2p[f] = pres
    test_pres = sorted({p2p[f] for f in split.test if f in p2p})
    out = []
    for pres in test_pres:
        drugs = [c for c in pres_drugs.get(pres, []) if c in seen]
        if len(drugs) >= 2:
            out.append({"id": pres, "drugs": drugs, "names": [names.get(c, str(c)) for c in drugs]})
        if len(out) >= n:
            break
    return out


def export_web_assets(web_dir: Path, detector: Path, embed: Path, prototypes: Path, params: Path,
                      root: Path | None = None, split_path: Path | None = None) -> dict:
    web_dir = Path(web_dir)
    (web_dir / "models").mkdir(parents=True, exist_ok=True)
    (web_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(detector, web_dir / "models" / "detector.int8.onnx")
    shutil.copy(embed, web_dir / "models" / "embed.int8.onnx")
    protos = load_prototypes_npz(prototypes)
    names = load_classes(root) if root else {}
    save_prototypes_json(web_dir / "data" / "prototypes.json", protos, names)
    seen = set(protos)
    aliases = load_aliases(root) if root else {}
    drugs = [{"id": int(c), "name": names.get(c, f"class_{c}"), "aliases": aliases.get(c, [])} for c in sorted(seen)]
    (web_dir / "data" / "drugs.json").write_text(json.dumps(drugs, ensure_ascii=False, indent=0), encoding="utf-8")
    cfg = SharedConfig().to_dict()
    cfg["decision"] = DecisionParams.from_dict(json.loads(Path(params).read_text())).to_dict()
    cfg["models"] = {"detector": "detector.int8.onnx", "embedding": "embed.int8.onnx",
                     "detector_bytes": Path(detector).stat().st_size, "embedding_bytes": Path(embed).stat().st_size}
    cfg["version"] = __version__
    cfg["built"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cfg["n_drugs"] = len(drugs)
    (web_dir / "data" / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    samples = []
    if root and split_path and Path(split_path).exists():
        samples = sample_prescriptions(root, Split.load(split_path), seen, names)
    (web_dir / "data" / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=0), encoding="utf-8")
    total_mb = (cfg["models"]["detector_bytes"] + cfg["models"]["embedding_bytes"]
                + (web_dir / "data" / "prototypes.json").stat().st_size) / 1e6
    print(f"web assets: {len(drugs)} drugs, {len(samples)} sample prescriptions, download {total_mb:.1f} MB")
    return cfg


def main(argv: list[str] | None = None) -> None:
    from pillguard.config import ARTIFACTS_DIR, VAIPE_DIR
    from pillguard.data.splits import default_split_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--web", type=Path, default=WEB_DIR)
    ap.add_argument("--detector", type=Path, default=ARTIFACTS_DIR / "export" / "detector.int8.onnx")
    ap.add_argument("--embed", type=Path, default=ARTIFACTS_DIR / "export" / "embed.int8.onnx")
    ap.add_argument("--prototypes", type=Path, default=ARTIFACTS_DIR / "decision" / "prototypes.npz")
    ap.add_argument("--params", type=Path, default=ARTIFACTS_DIR / "decision" / "params.json")
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    a = ap.parse_args(argv)
    export_web_assets(a.web, a.detector, a.embed, a.prototypes, a.params, a.root, a.split)


if __name__ == "__main__":
    main()
