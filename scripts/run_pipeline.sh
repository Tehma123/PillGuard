#!/usr/bin/env bash
# End-to-end training + export + evaluation after `pillguard ingest`. Re-runnable; each stage
# writes under artifacts/. Usage: bash scripts/run_pipeline.sh [det_epochs] [emb_epochs]
# (Git Bash on Windows works; PowerShell users: scripts/run_pipeline.ps1). About 80-90 minutes on an RTX 3060 (measured: detector 2.3 min/epoch, embedding 1.4 min/epoch).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
DET_EPOCHS="${1:-30}"; EMB_EPOCHS="${2:-20}"; DET_EXTRA="${DET_EXTRA:-}"   # e.g. DET_EXTRA="--cache ram --workers 8"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "classes / split / scenarios / yolo dataset"
python -m pillguard.cli classes
python -m pillguard.cli split            # no-op if splits/vaipe_v1.json exists
python -m pillguard.cli scenarios
python -m pillguard.cli det-prepare

log "crop cache (CPU) and detector training (GPU) in parallel"
python -m pillguard.cli crops --workers 6 > artifacts/crops.log 2>&1 &
CROPS_PID=$!
python -m pillguard.cli det-train --data data/vaipe/yolo/pill.yaml --model artifacts/yolo11n.pt \
  --epochs "$DET_EPOCHS" --batch 32 --name yolo11n $DET_EXTRA > artifacts/det_train.log 2>&1 &
DET_PID=$!
wait $CROPS_PID; log "crops done"

log "embedding training (GPU, shares the card with the detector if still running)"
python -m pillguard.cli emb-train --epochs "$EMB_EPOCHS" --batch 128 --workers 4 > artifacts/emb_train.log 2>&1
log "embedding export + int8"
python -m pillguard.cli emb-export --ckpt artifacts/embed/mnv3s/best.pt > artifacts/emb_export.log 2>&1
log "prototypes + calibration + thresholds"
python -m pillguard.cli fit --embed-onnx artifacts/export/embed.int8.onnx --ckpt artifacts/embed/mnv3s/best.pt > artifacts/fit.log 2>&1

wait $DET_PID; log "detector done"
python -m pillguard.cli det-export --weights artifacts/detector/yolo11n/weights/best.pt \
  --calib-list data/vaipe/yolo/val.txt > artifacts/det_export.log 2>&1

log "web assets + evaluation"
python -m pillguard.cli export-web
python -m pillguard.cli eval --mode detector > artifacts/eval_detector.log 2>&1
python -m pillguard.cli eval --mode oracle > artifacts/eval_oracle.log 2>&1
python -m pillguard.cli eda

log "python vs browser parity (needs node + npm install in web/)"
python -m pillguard.cli parity --export > artifacts/parity.log 2>&1
(cd web && node parity_node.mjs ../artifacts/parity . > ../artifacts/parity_node.log 2>&1)
python -m pillguard.cli parity --compare
python scripts/fill_readme.py
log "PIPELINE DONE - see artifacts/eval/test_detector/report.md and README.md"
