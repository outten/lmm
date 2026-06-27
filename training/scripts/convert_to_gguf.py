"""Convert a Hugging Face checkpoint to GGUF for Ollama.

Two paths:
  - merged/  (full precision model saved by sft_train.py --save-merged)
  - adapter/ (LoRA adapter + base model id, merged at conversion time)

Uses llama.cpp's convert script. After conversion, run:
    ollama create outten/lmm:TAG -f Modelfile.TAG

Usage:
    python training/scripts/convert_to_gguf.py training/runs/v0.1/merged \\
        --out training/runs/v0.1/gguf \\
        --quant Q4_K_M
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LLAMACPP_REPO = "https://github.com/ggerganov/llama.cpp"


def find_convert_script() -> Path | None:
    """Look for llama.cpp's convert_hf_to_gguf.py in common locations."""
    candidates = [
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path.home() / "src" / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("model_dir", type=Path,
                   help="HF model directory (merged or adapter+base)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for GGUF files")
    p.add_argument("--quant", default="Q4_K_M",
                   help="Quantization (Q4_K_M, Q5_K_M, Q8_0, F16, F32, ...)")
    p.add_argument("--llama-cpp-dir", type=Path, default=None,
                   help="Path to a llama.cpp checkout. Auto-detected if omitted.")
    args = p.parse_args()

    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    llama_cpp = args.llama_cpp_dir or find_convert_script() and find_convert_script().parent
    if llama_cpp is None:
        print("llama.cpp not found.", file=sys.stderr)
        print(f"Clone with: git clone {LLAMACPP_REPO}", file=sys.stderr)
        print("Then re-run with --llama-cpp-dir.", file=sys.stderr)
        return 2

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_script = llama_cpp / "quantize"
    if not convert_script.exists():
        print(f"convert_hf_to_gguf.py not found in {llama_cpp}", file=sys.stderr)
        return 3
    if not quantize_script.exists():
        print(f"quantize binary not found in {llama_cpp} (run 'make' in llama.cpp)", file=sys.stderr)
        return 4

    # Step 1: HF -> GGUF (f16)
    f16_out = args.out / "model.f16.gguf"
    print(f"[gguf] Converting {args.model_dir} -> {f16_out}")
    subprocess.check_call([
        sys.executable, str(convert_script),
        str(args.model_dir),
        "--outfile", str(f16_out),
        "--outtype", "f16",
    ])

    # Step 2: quantize
    quant_out = args.out / f"model.{args.quant.lower()}.gguf"
    print(f"[gguf] Quantizing -> {quant_out}")
    subprocess.check_call([str(quantize_script), str(f16_out), str(quant_out), args.quant])

    print(f"[gguf] Done. Quantized model at: {quant_out}")
    print(f"[gguf] Next: ollama create outten/lmm:TAG -f Modelfile.TAG")
    return 0


if __name__ == "__main__":
    sys.exit(main())