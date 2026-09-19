"""Download the VAIPE mirror from Hugging Face and store it in a compact local layout.

The public mirror ``Elfsong/VAIPE_PILL`` is the AI4VN-2022 challenge release:

* ``pill``          - 9,502 labelled photos (public_train) + 1,500 unlabelled (public_test),
                      4032x3024 JPEGs with ``bboxes`` as ``[x, y, w, h]`` and ``labels`` 0..107
                      (107 == "pill not in this prescription").
* ``prescription``  - 1,173 (+172 unlabelled) prescription scans with OCR text lines, line labels
                      (``drugname``, ``quantity``, ...) and a mapping line -> pill class id.
* ``pill_pres_map`` - which pill photos belong to which prescription.

Full-resolution photos are ~27 GB, far more than any model here needs, so every photo is
resized on the fly to ``INGEST_MAX_SIDE`` and boxes are rescaled with it. Output layout::

    data/vaipe/
      images/pill/<file>.jpg
      images/prescription/<file>.jpg
      annotations/pill_train.jsonl        one JSON object per photo
      annotations/prescription_train.jsonl
      pill_pres_map_train.json            {prescription_file: [pill_file, ...]}
      ingest_state.json                   which shards are done (resumable)

The ``test`` split of the mirror has no labels at all, so it is skipped by default.

Run: ``python -m pillguard.data.ingest [--configs pill prescription pill_pres_map] [--limit N]``
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

from pillguard.config import HF_DATASET, INGEST_MAX_SIDE, PRESCRIPTION_MAX_SIDE, VAIPE_DIR

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few VAIPE JPEGs miss their final byte
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------
_EXIF_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def decode_resized(data: bytes, max_side: int) -> tuple[Image.Image, tuple[int, int], float, int]:
    """Decode JPEG/PNG bytes, apply the EXIF orientation, shrink so that max(w, h) == max_side.

    VAIPE boxes were drawn on the photo *as displayed* (EXIF-rotated), so the rotation must
    be applied before the boxes are used. Uses JPEG DCT-domain draft decoding for speed.
    Returns (image, displayed_original_size, scale, orientation_tag).
    """
    im = Image.open(io.BytesIO(data))
    try:
        orientation = int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        orientation = 1
    sw, sh = im.size                                   # stored pixel grid
    rotated = orientation in (5, 6, 7, 8)
    ow, oh = (sh, sw) if rotated else (sw, sh)         # displayed size (annotation frame)
    scale = min(1.0, max_side / float(max(ow, oh)))
    tw, th = max(1, round(ow * scale)), max(1, round(oh * scale))
    if scale < 1.0 and im.format == "JPEG":
        im.draft("RGB", (th, tw) if rotated else (tw, th))  # largest power-of-2 reduction >= target
    im = im.convert("RGB")
    if orientation in _EXIF_TRANSPOSE:
        im = im.transpose(_EXIF_TRANSPOSE[orientation])
    if im.size != (tw, th):
        im = im.resize((tw, th), Image.BILINEAR)
    return im, (ow, oh), scale, orientation


def xywh_to_xyxy_scaled(boxes: list[list[int]], scale: float, w: int, h: int) -> list[list[float]]:
    out = []
    for x, y, bw, bh in boxes:
        x1 = max(0.0, min(w, x * scale))
        y1 = max(0.0, min(h, y * scale))
        x2 = max(0.0, min(w, (x + bw) * scale))
        y2 = max(0.0, min(h, (y + bh) * scale))
        out.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    return out


def save_jpeg(im: Image.Image, path: Path) -> None:
    tmp = path.with_name(path.stem + ".tmp.jpg")
    im.save(tmp, "JPEG", quality=90, optimize=True)
    os.replace(tmp, path)


def jpg_name(filename: str) -> str:
    return Path(filename).stem + ".jpg"


# --------------------------------------------------------------------------------------
# Row processors (one per HF config)
# --------------------------------------------------------------------------------------
def process_pill_row(row: dict, split: str, img_dir: Path) -> dict:
    fname = jpg_name(row["filename"])
    out_path = img_dir / fname
    im, (ow, oh), scale, orientation = decode_resized(row["image"]["bytes"], INGEST_MAX_SIDE)
    w, h = im.size
    if not out_path.exists():
        save_jpeg(im, out_path)
    boxes = xywh_to_xyxy_scaled(row["bboxes"] or [], scale, w, h)
    labels = list(row["labels"] or [])
    keep = [i for i, b in enumerate(boxes) if (b[2] - b[0]) >= 2 and (b[3] - b[1]) >= 2]
    return {
        "file": fname,
        "orig_file": row["filename"],
        "src_split": split,
        "width": w,
        "height": h,
        "orig_size": [ow, oh],
        "exif_orientation": orientation,
        "boxes": [boxes[i] for i in keep],
        "labels": [labels[i] for i in keep],
        "n_dropped_boxes": len(boxes) - len(keep),
    }


def process_prescription_row(row: dict, split: str, img_dir: Path) -> dict:
    fname = jpg_name(row["filename"])
    out_path = img_dir / fname
    im, (ow, oh), scale, orientation = decode_resized(row["image"]["bytes"], PRESCRIPTION_MAX_SIDE)
    w, h = im.size
    if not out_path.exists():
        save_jpeg(im, out_path)
    return {
        "file": fname,
        "orig_file": row["filename"],
        "src_split": split,
        "width": w,
        "height": h,
        "orig_size": [ow, oh],
        "exif_orientation": orientation,
        "scale": scale,
        "ann_ids": list(row["ann_ids"] or []),
        "texts": list(row["texts"] or []),
        "ann_labels": list(row["ann_labels"] or []),
        # kept exactly as in the source (unscaled); multiply by `scale` to draw on the saved image
        "boxes_raw": [list(b) for b in (row["boxes"] or [])],
        "mappings": list(row["mappings"] or []),
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
class Ingestor:
    def __init__(self, out_dir: Path = VAIPE_DIR, keep_shards: bool = False, workers: int = 6,
                 limit: int | None = None, splits: tuple[str, ...] = ("train",)) -> None:
        self.out_dir = Path(out_dir)
        self.cache_dir = self.out_dir / "_hfcache"
        self.keep_shards = keep_shards
        self.workers = workers
        self.limit = limit
        self.splits = splits
        (self.out_dir / "images" / "pill").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images" / "prescription").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "annotations").mkdir(parents=True, exist_ok=True)
        self.state_path = self.out_dir / "ingest_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"done": []}

    # -- state ---------------------------------------------------------------------------
    def _mark_done(self, shard: str) -> None:
        self.state["done"].append(shard)
        self.state_path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def _shards(self, config: str, split: str) -> list[str]:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(HF_DATASET, repo_type="dataset")
        prefix = f"{config}/{split}-"
        return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))

    def _download(self, shard: str) -> Path:
        from huggingface_hub import hf_hub_download

        last: Exception | None = None
        for attempt in range(6):
            try:
                return Path(hf_hub_download(HF_DATASET, shard, repo_type="dataset",
                                            cache_dir=str(self.cache_dir)))
            except Exception as e:  # network hiccup: back off and retry
                last = e
                wait = 10 * (attempt + 1)
                print(f"  download failed ({e}); retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"could not download {shard}: {last}")

    def _cleanup(self, path: Path) -> None:
        if self.keep_shards:
            return
        try:
            path.unlink()
        except OSError:
            pass

    # -- configs -------------------------------------------------------------------------
    def ingest_images(self, config: str, split: str) -> None:
        import pyarrow.parquet as pq

        proc = process_pill_row if config == "pill" else process_prescription_row
        img_dir = self.out_dir / "images" / config
        ann_path = self.out_dir / "annotations" / f"{config}_{split}.jsonl"
        shards = self._shards(config, split)
        if self.limit:
            shards = shards[: self.limit]
        todo = [s for s in shards if s not in self.state["done"]]
        print(f"[{config}/{split}] {len(shards)} shards, {len(todo)} to do", flush=True)
        t0 = time.time()
        for i, shard in enumerate(todo):
            path = self._download(shard)
            table = pq.read_table(path)
            rows = table.to_pylist()
            with ThreadPoolExecutor(self.workers) as ex:
                records = list(ex.map(lambda r: proc(r, split, img_dir), rows))
            with open(ann_path, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
            del table, rows
            self._cleanup(path)
            self._mark_done(shard)
            el = time.time() - t0
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"  {i + 1}/{len(todo)} {shard.split('/')[-1]} rows={len(records)} "
                  f"elapsed={el / 60:.1f}m eta={eta / 60:.1f}m", flush=True)

    def ingest_map(self, split: str) -> None:
        import pyarrow.parquet as pq

        out_path = self.out_dir / f"pill_pres_map_{split}.json"
        mapping: dict[str, list[str]] = {}
        for shard in self._shards("pill_pres_map", split):
            path = self._download(shard)
            for r in pq.read_table(path).to_pylist():
                mapping.setdefault(jpg_name(r["prescription"]), []).append(jpg_name(r["pill"]))
            self._cleanup(path)
        for k in mapping:
            mapping[k] = sorted(set(mapping[k]))
        out_path.write_text(json.dumps(mapping, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[pill_pres_map/{split}] {len(mapping)} prescriptions -> {out_path}", flush=True)

    def run(self, configs: list[str]) -> None:
        for split in self.splits:
            if "pill_pres_map" in configs:
                self.ingest_map(split)
            if "prescription" in configs:
                self.ingest_images("prescription", split)
            if "pill" in configs:
                self.ingest_images("pill", split)
        print("INGEST DONE", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=VAIPE_DIR)
    ap.add_argument("--configs", nargs="+", default=["pill_pres_map", "prescription", "pill"])
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="mirror splits to ingest; 'test' is unlabelled and skipped by default")
    ap.add_argument("--limit", type=int, default=None, help="only the first N shards per split (smoke test)")
    ap.add_argument("--keep-shards", action="store_true", help="keep downloaded parquet files")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    Ingestor(a.out, keep_shards=a.keep_shards, workers=a.workers, limit=a.limit,
             splits=tuple(a.splits)).run(a.configs)


if __name__ == "__main__":
    main()
