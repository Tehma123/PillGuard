"""ONNX export, simplification and INT8 quantisation helpers shared by both models.

Quantisation is *static* QDQ (uint8 activations, per-channel int8 weights) calibrated on a
few dozen real inputs. This is the format ONNX Runtime Web executes with integer kernels in
WebAssembly; dynamic quantisation would leave the convolutions in fp32.

Both models exclude a few nodes from quantisation and are therefore mixed precision, because
one uint8 scale per tensor destroys them otherwise; see ``nodes_to_exclude`` at each call
site. Callers are expected to compare the result against fp32 and raise
:class:`QuantizationError` when it degrades.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


class QuantizationError(RuntimeError):
    """An INT8 model disagrees with its fp32 source beyond the accepted margin."""


def onnx_size_mb(path: Path) -> float:
    return os.path.getsize(path) / 1e6


def export_torch(model, example: torch.Tensor, path: Path, input_name: str = "input",
                 output_name: str = "output", opset: int = 17, dynamic_batch: bool = True) -> Path:
    import torch

    model.eval()
    dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(model, example, str(path), input_names=[input_name], output_names=[output_name],
                      opset_version=opset, dynamic_axes=dynamic_axes, do_constant_folding=True, dynamo=False)
    simplify_onnx(path)
    return path


def simplify_onnx(path: Path) -> Path:
    try:
        import onnx
        import onnxslim

        model = onnxslim.slim(onnx.load(str(path)))
        onnx.save(model, str(path))
    except Exception as e:  # optional dependency
        print(f"  (onnxslim skipped: {e})")
    return path


class _ArrayReader:
    """CalibrationDataReader over an in-memory list of input arrays."""

    def __init__(self, input_name: str, arrays: list[np.ndarray]) -> None:
        self.input_name = input_name
        self.arrays = arrays
        self.i = 0

    def get_next(self):
        if self.i >= len(self.arrays):
            return None
        a = self.arrays[self.i]
        self.i += 1
        return {self.input_name: a.astype(np.float32)}

    def rewind(self):
        self.i = 0


def quantize_static_int8(fp32_path: Path, int8_path: Path, calibration: list[np.ndarray], input_name: str,
                         per_channel: bool = True, method: str = "minmax", nodes_to_exclude: list[str] | None = None,
                         op_types: list[str] | None = None) -> Path:
    """Static QDQ INT8 quantisation calibrated with ``calibration`` (list of NCHW float arrays)."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    methods = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
               "percentile": CalibrationMethod.Percentile}
    with tempfile.TemporaryDirectory() as td:
        pre = Path(td) / "pre.onnx"
        try:
            quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        except Exception as e:
            print(f"  (quant_pre_process skipped: {e})")
            shutil.copy(fp32_path, pre)
        extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
        if method == "percentile":
            extra["CalibPercentile"] = 99.99
        elif method == "minmax":
            extra["CalibMovingAverage"] = True
        quantize_static(
            str(pre), str(int8_path), _ArrayReader(input_name, calibration),
            quant_format=QuantFormat.QDQ, per_channel=per_channel, reduce_range=False,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=methods[method], nodes_to_exclude=nodes_to_exclude or [],
            op_types_to_quantize=op_types, extra_options=extra,
        )
    print(f"  int8: {onnx_size_mb(fp32_path):.1f} MB -> {onnx_size_mb(int8_path):.1f} MB ({int8_path.name})")
    return int8_path


def ort_session(path: Path, threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(session, x: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    return session.run(None, {name: x.astype(np.float32)})[0]
