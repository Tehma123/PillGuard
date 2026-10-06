# PillGuard

> Prescription-aware pill checking with a reject option, running entirely in the browser.

PillGuard takes a prescription (a drug list) and a photo of the pills in your hand, then flags
every pill that is **not on the prescription**, or says **"not sure"** and asks you to retake
the photo or ask a pharmacist. Nothing leaves the device: detection, recognition and the
decision run in the browser with ONNX Runtime Web.

> **Research demo, not a medical device.** No dosage advice. Evaluated on one public dataset
> of 107 Vietnamese drugs; anything else counts as "not on the prescription".

**Demo:** https://tehma123.github.io/PillGuard/ · **Specifications:** [Specifications.md](Specifications.md) ·
**Data provenance, licensing and the week-0 experiments:** [Specifications.md §10](Specifications.md#10-resolved-questions-and-evidence)

## Results

Test split: 1,449 photos / 5,034 pills from 200 prescriptions never seen in training
(`splits/vaipe_v1.json`). Out-of-prescription pills come from three sources: drugs deleted
from the prescription (the Specifications §3.5 scenario), pills VAIPE itself labels as foreign, and
drugs the model never trained on ("unseen": 12 held out on purpose, plus a few rare ones that no
training photo contains). Each photo is checked against its prescription as written and with one or
two drugs deleted, so pill-level rates count a pill once per check.

<!-- RESULTS:START -->
| metric (test split) | PillGuard (with reject) | same model, no reject | SPEC target |
|---|---|---|---|
| out-of-prescription recall | **98.8 %** | 95.4 % | ≥ 95 % |
| false alarm rate (listed pill flagged) | **1.1 %** | 4.5 % | ≤ 10 % |
| abstain rate | **17.2 %** | 0 % | ≤ 20 % |
| error rate on decided pills (risk) | **1.1 %** | 4.5 % | lower with reject |
| out recall, strict (misses and abstains count as missed) | 80.7 % | 95.4 % | – |
| detection recall (pill found at IoU ≥ 0.5) | 99.6 % | same | – |
| ECE of p(in prescription) | 0.009 | – | lower than uncalibrated |
| **per photo:** only prescribed pills, yet a pill flagged OUT (false alert) | **2.0 %** | – | not in SPEC |
| per photo: only prescribed pills, a pill *not sure* and none OUT | 37.6 % | – | not in SPEC |
| per photo: a wrong pill present, at least one pill flagged OUT | 92.5 % | – | not in SPEC |
| per photo: a wrong pill present, nothing flagged at all | 0.3 % | – | not in SPEC |
| ECE on validation, before → after calibration | 0.071 → 0.004 | – | – |
| detector mAP@0.5 / mAP@0.5:0.95 (test) | 0.992 / 0.759 | – | set after baseline |
| model download (INT8 ONNX) | 4.9 MB (detector 3.3 + embedding 1.7) | – | ≤ 30 MB |
| Python ONNX latency, 1 thread (detect + embed) | median 85 ms, p90 93 ms | – | ≤ 1 s in browser |
| Python vs browser (Node/WASM) verdict agreement | 100 % on 40 cases | – | 100 % |

| pills | pill checks | out recall | false alarm | abstain | out pill accepted |
|---|---|---|---|---|---|
| on the prescription | 5787 | – | 1.1 % | 16.6 % | – |
| deleted from the prescription | 3012 | 99.6 % | – | 7.4 % | 0.4 % |
| drugs never trained on (7 of the 12 held out, 4 with no train photo) | 349 | 100.0 % | – | 21.8 % | 0.0 % |
| labelled foreign by VAIPE (drug unknown) | 1475 | 96.1 % | – | 38.7 % | 2.4 % |

With ground-truth boxes (recognition + decision only): out recall 98.6 %, false alarms 1.1 %, abstain 17.4 %, ECE 0.011.

Full tables, calibration and quantisation detail, risk-coverage and reliability plots: [docs/benchmark.md](docs/benchmark.md). Confusion pairs with example crops are VAIPE photographs and stay out of the repository; `pillguard eval` writes them under `artifacts/eval/`.
<!-- RESULTS:END -->

## How it works

```
prescription (drug ids)  ─────────────────────────────┐
photo ─► YOLO11n (1 class: pill) ─► 128 px crops ─► MobileNetV3-S embedding ─► match ─► in / out / uncertain
```

1. **Detector.** YOLO11n trained on every box as one class. Recognition is *not* done by
   the detector, so adding a drug never touches it.
2. **Embedding network.** MobileNetV3-Small + projection to 128-d, trained with an ArcFace
   loss over the seen drugs. At inference the classifier weights are thrown away.
3. **Prototypes.** Each drug's reference is the mean embedding of its training crops plus two
   k-means sub-centres (VAIPE ships no reference photos, see [Specifications.md §10.3](Specifications.md#103-where-do-the-per-drug-reference-images-come-from--cropped-from-training-photos)).
4. **Decision.** Cosine similarity to every drug, softmax with a fitted temperature and a
   virtual *unknown* class, `p_in = Σ p(listed drugs)`. Two thresholds chosen on validation
   under caps (false alarms ≤ 10 %, abstentions ≤ 20 %) give `in / out / uncertain`.
   The same arithmetic runs in [`web/pipeline.js`](web/pipeline.js); a parity test checks that
   Python and the browser give identical verdicts on identical pixels.
5. **Prescription input** (Specifications §5.6 options): pick drugs from the list (C), load a sample
   prescription (B), or scan a prescription photo with in-browser Vietnamese OCR
   (A, experimental; see the week-0 experiment in [Specifications.md §10.5](Specifications.md#105-week-0-experiment-in-browser-prescription-ocr-settling-paths-abc)).

Everything the browser needs is under `web/`: two INT8 ONNX models, `prototypes.json`,
`drugs.json`, `config.json`. No build step; GitHub Pages serves the folder as is.

## Reproduce

Requirements: Python ≥ 3.10, an NVIDIA GPU for training (an RTX 3060 12 GB was used), Node 22
for the browser parity test. Full-resolution VAIPE is 27 GB; the ingest shrinks it to 1.6 GB.

One command runs everything after the ingest (`scripts/run_pipeline.sh` for bash/Git Bash,
`scripts/run_pipeline.ps1` for PowerShell; about 80-90 minutes on an RTX 3060 with the defaults of
30 detector epochs and 20 embedding epochs). Stage by stage:

```bash
pip install -e ".[dev,data]"

pillguard ingest                      # Hugging Face mirror -> data/vaipe (25 min at 18 MB/s)
pillguard classes && pillguard split  # class table, fixed split (already committed under splits/)
pillguard scenarios && pillguard eda  # out-of-prescription scenarios, docs/eda.md

pillguard det-prepare
pillguard det-train --data data/vaipe/yolo/pill.yaml --model artifacts/yolo11n.pt --epochs 30   # artifacts/detector/yolo11n
pillguard det-export --weights artifacts/detector/yolo11n/weights/best.pt --calib-list data/vaipe/yolo/val.txt

pillguard crops
pillguard emb-train --epochs 20                                   # artifacts/embed/mnv3s
pillguard emb-export --ckpt artifacts/embed/mnv3s/best.pt
pillguard fit --embed-onnx artifacts/export/embed.int8.onnx --ckpt artifacts/embed/mnv3s/best.pt

pillguard export-web                  # copies models + data into web/
pillguard eval --mode detector        # artifacts/eval/test_detector/report.md
pillguard eval --mode oracle          # ground-truth boxes: recognition + decision only

pillguard parity --export && (cd web && npm ci && npm run parity) && pillguard parity --compare
python scripts/fill_readme.py         # results table above + docs/benchmark.md, from artifacts/
pytest                                # unit tests + JS/Python parity on synthetic data (no dataset needed)
```

`python -m http.server -d web 8080` serves the demo locally.

## Repository layout

| path | what |
|---|---|
| `pillguard/data/` | VAIPE ingest (EXIF-aware), loaders, fixed split, scenario generator, synthetic data, EDA |
| `pillguard/detector/` | YOLO dataset prep, training, ONNX + INT8 export |
| `pillguard/embed/` | crop cache, embedding net + ArcFace, training, prototypes, export |
| `pillguard/decision/` | matcher, temperature/unknown-bias calibration, threshold search, fit script |
| `pillguard/eval/` | metrics, full-pipeline evaluation, plots, Python-vs-browser parity |
| `pillguard/imaging.py` | preprocessing shared with the browser (bilinear resize, letterbox, crops, NMS) |
| `pillguard/pipeline.py` | reference ONNX Runtime pipeline |
| `web/` | the demo: `index.html`, `app.js`, `pipeline.js`, `ocr_match.js`, models and data |
| `splits/vaipe_v1.json` | the fixed split (file names + held-out drugs) |
| `docs/` | `eda.md` (dataset), `benchmark.md` (results), `eda_stats.json`, figures |
| `tests/` | pytest suite; runs on synthetic data, including the Node parity test |

## Data

VAIPE (VinUni-Illinois Smart Health Center), via the public Hugging Face mirror
`Elfsong/VAIPE_PILL` = the AI4VN-2022 challenge release: 9,502 labelled photos, 32,828 pills,
107 drugs, 1,173 prescriptions with OCR lines mapped to drug ids. The mirror's test split is
unlabelled, so the split here is our own, grouped by prescription. Two things worth knowing:

* Boxes were annotated on the EXIF-rotated photo; the ingest applies the orientation tag first.
* 8,398 pills carry label 107 ("not on this prescription"), a ready-made out-of-prescription set.

No dataset content is redistributed here: `data/` and `artifacts/` are git-ignored (raw shards,
photos, annotations, crops, evaluation records). The repository carries only code, the split
file (photo names), aggregate statistics, and the trained INT8 weights and prototype vectors
the demo needs. Licensing evidence is collected in [Specifications.md §10.2](Specifications.md#102-do-the-data-terms-allow-publishing-models-and-a-public-demo--no-licence-text-exists-proceeding-on-documented-precedent).

## License

Code: AGPL-3.0 (the detector is built with Ultralytics, which is AGPL-3.0). Trained weights
under `web/models/` are derived from VAIPE and are provided for research use.

## Acknowledgements

VAIPE dataset and the PGPNet paper (Nguyen et al., PLOS ONE 2023); Ultralytics YOLO;
ONNX Runtime Web; Tesseract.js.
