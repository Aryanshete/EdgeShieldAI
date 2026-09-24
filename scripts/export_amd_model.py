"""Export and optimize YOLOv8 model for AMD ROCm and Ryzen AI execution targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.amd_backend import DEFAULT_MODEL_PATH, DEFAULT_ONNX_PATH, export_yolo_to_onnx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 PyTorch model to ONNX format optimized for AMD ROCm / Ryzen AI."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to input PyTorch .pt model file (default: models/yolov8n.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Path for exported .onnx model file (default: models/yolov8n.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX operator set version (default: 17 for AMD Vitis AI / ROCm compatibility)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Enable dynamic input shapes for variable surveillance resolutions",
    )
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        help="Disable dynamic input shapes (fixed 640x640)",
    )

    args = parser.parse_args()

    print(f"[*] Exporting {args.model} to ONNX (Opset {args.opset}, Dynamic: {args.dynamic})...")
    try:
        exported_path = export_yolo_to_onnx(
            model_path=args.model,
            output_path=args.output,
            opset=args.opset,
            dynamic=args.dynamic,
        )
        print(f"[+] Successfully exported ONNX model to: {exported_path}")

        # Validate with onnx and onnxruntime
        import onnx

        model_proto = onnx.load(str(exported_path))
        onnx.checker.check_model(model_proto)
        print(f"[+] ONNX model syntax and topological check passed (IR v{model_proto.ir_version})")

        import onnxruntime as ort

        sess = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        print(f"[+] ONNX Runtime session initialized successfully:")
        print(f"    Inputs:  {[i.name for i in inputs]} (shape: {[i.shape for i in inputs]})")
        print(f"    Outputs: {[o.name for o in outputs]} (shape: {[o.shape for o in outputs]})")

        return 0
    except Exception as exc:
        print(f"[-] Model export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
