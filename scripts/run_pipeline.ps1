# End-to-end training + export + evaluation after `pillguard ingest` (Windows PowerShell 5.1+).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 [-DetEpochs 30] [-EmbEpochs 20]
# About 35 minutes on an RTX 3060 once the crop cache exists (measured over 30 + 20 epochs: detector
# 0.85 min/epoch, embedding 1.09 min/epoch, the two sharing the card); the first run also writes
# 2 x 32828 crops. Each stage logs under artifacts\ and can be re-run alone
# with the matching `python -m pillguard.cli ...` command.
param([int]$DetEpochs = 30, [int]$EmbEpochs = 20, [string]$DetExtra = "")   # e.g. -DetExtra "--cache ram --workers 8"
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force artifacts | Out-Null
function Log($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $m) }
function Run($args, $log) {
    $p = Start-Process -FilePath python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if ($p.ExitCode -ne 0) { throw "failed: python $args (see $log, $log.err)" }
}

Log "classes / split / scenarios / yolo dataset"
python -m pillguard.cli classes; if ($LASTEXITCODE) { throw "classes failed" }
python -m pillguard.cli split
python -m pillguard.cli scenarios; if ($LASTEXITCODE) { throw "scenarios failed" }
python -m pillguard.cli det-prepare; if ($LASTEXITCODE) { throw "det-prepare failed" }

Log "crop cache (CPU) and detector training (GPU) in parallel"
$crops = Start-Process -FilePath python -ArgumentList "-m pillguard.cli crops --workers 6" -NoNewWindow -PassThru -RedirectStandardOutput artifacts\crops.log -RedirectStandardError artifacts\crops.err
$det = Start-Process -FilePath python -ArgumentList "-m pillguard.cli det-train --data data/vaipe/yolo/pill.yaml --model artifacts/yolo11n.pt --epochs $DetEpochs --batch 32 --name yolo11n $DetExtra" -NoNewWindow -PassThru -RedirectStandardOutput artifacts\det_train.log -RedirectStandardError artifacts\det_train.err
$crops.WaitForExit(); if ($crops.ExitCode -ne 0) { throw "crops failed (artifacts\crops.err)" }; Log "crops done"

Log "embedding training (GPU; shares the card with the detector while it runs)"
Run "-m pillguard.cli emb-train --epochs $EmbEpochs --batch 128 --workers 4" artifacts\emb_train.log
Log "embedding export + int8"
Run "-m pillguard.cli emb-export --ckpt artifacts/embed/mnv3s/best.pt" artifacts\emb_export.log
Log "prototypes + calibration + thresholds"
Run "-m pillguard.cli fit --embed-onnx artifacts/export/embed.int8.onnx --ckpt artifacts/embed/mnv3s/best.pt" artifacts\fit.log

$det.WaitForExit(); if ($det.ExitCode -ne 0) { throw "detector training failed (artifacts\det_train.err)" }; Log "detector done"
Run "-m pillguard.cli det-export --weights artifacts/detector/yolo11n/weights/best.pt --calib-list data/vaipe/yolo/val.txt" artifacts\det_export.log

Log "web assets + evaluation"
Run "-m pillguard.cli export-web" artifacts\export_web.log
Run "-m pillguard.cli eval --mode detector" artifacts\eval_detector.log
Run "-m pillguard.cli eval --mode oracle" artifacts\eval_oracle.log
Run "-m pillguard.cli eda" artifacts\eda.log

Log "python vs browser parity (needs node + `npm install` in web\)"
Run "-m pillguard.cli parity --export" artifacts\parity.log
Push-Location web; node parity_node.mjs ..\artifacts\parity . | Out-File -Encoding utf8 ..\artifacts\parity_node.log; Pop-Location
Run "-m pillguard.cli parity --compare" artifacts\parity_compare.log
python scripts\fill_readme.py
Log "PIPELINE DONE - see artifacts\eval\test_detector\report.md and README.md"
