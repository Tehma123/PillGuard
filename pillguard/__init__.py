"""PillGuard: prescription-aware pill checking with a reject option.

Layout
------
- ``pillguard.data``      : VAIPE ingestion, fixed splits, out-of-prescription scenarios, synthetic data
- ``pillguard.detector``  : single-class pill detector (Ultralytics YOLO) + ONNX/INT8 export
- ``pillguard.embed``     : lightweight embedding network, metric-learning training, class prototypes
- ``pillguard.decision``  : matching against the prescription, calibration, reject thresholds
- ``pillguard.eval``      : metrics (out-of-prescription recall, abstention, risk-coverage, ECE), reports
- ``pillguard.pipeline``  : end-to-end ONNX Runtime inference that mirrors the browser code
"""

__version__ = "0.1.0"
