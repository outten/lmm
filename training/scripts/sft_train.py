"""Supervised fine-tuning for outten/lmm using Unsloth + LoRA.

Trains a base model (e.g. Qwen2.5-Math-7B-Instruct) on the code-first
response format documented in PROPOSAL.md §13. Uses QLoRA by default
so 7B-class models fit on a single A100/H100 (80GB).

Usage:
    python training/scripts/sft_train.py \
        --base Qwen/Qwen2.5-Math-7B-Instruct \
        --data training/data/sft.jsonl \
        --output training/runs/v0.1 \
        --epochs 2 --lr 2e-5 --batch-size 4

The output is a Hugging Face checkpoint directory ready for GGUF
conversion:
    python training/scripts/convert_to_gguf.py training/runs/v0.1

Then build the Ollama model:
    ollama create outten/lmm:v0.1 -f Modelfile.v0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Heavy imports are inside main() so the script can be --help-checked
# without a GPU.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--base", required=True,
                   help="HF model id (e.g. Qwen/Qwen2.5-Math-7B-Instruct)")
    p.add_argument("--data", required=True, type=Path,
                   help="jsonl with {prompt, response} per line")
    p.add_argument("--output", required=True, type=Path,
                   help="Output checkpoint directory")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--qlora", action="store_true", default=True,
                   help="Use 4-bit quantization (QLoRA). Default on.")
    p.add_argument("--no-qlora", dest="qlora", action="store_false")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap training set size (debugging)")
    p.add_argument("--save-merged", action="store_true",
                   help="Save merged full-precision model in addition to adapter")
    return p.parse_args()


def load_dataset(path: Path, max_samples: int | None):
    from datasets import load_dataset
    # jsonl with {prompt, response}; SFTTrainer accepts a dataset
    # where each row has a "text" column (we'll assemble it) or any
    # format string. We use the chat-template format directly.
    ds = load_dataset("json", data_files=str(path), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def to_text(ex):
        # Format that matches PROPOSAL.md §13 response shape.
        # The model sees prompt + response as one continuous stream;
        # during inference only the prompt is sent and the model
        # continues with the response.
        return {
            "text": f"{ex['prompt']}\n\n{ex['response']}",
        }
    return ds.map(to_text, remove_columns=ds.column_names)


def main() -> int:
    args = parse_args()

    # Heavy imports
    try:
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install with:", file=sys.stderr)
        print("  pip install -U unsloth transformers trl peft accelerate datasets bitsandbytes", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("CUDA not available. Training requires a GPU.", file=sys.stderr)
        print("On Apple Silicon, see training/scripts/mlx_train.py for an alternative.", file=sys.stderr)
        return 3

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[sft] Loading base model: {args.base}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.qlora,
        dtype=None,  # autodetect (bf16/fp16)
    )

    print(f"[sft] Adding LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(f"[sft] Loading dataset: {args.data}")
    ds = load_dataset(args.data, max_samples=args.max_samples)
    print(f"[sft] {len(ds)} training examples")

    sft_config = SFTConfig(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=ds,
    )

    print("[sft] Training...")
    trainer.train()

    print(f"[sft] Saving adapter to {args.output}")
    trainer.save_model(str(args.output))

    if args.save_merged:
        print("[sft] Saving merged model (16-bit) for GGUF conversion")
        merged_dir = args.output / "merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"[sft] Merged model at {merged_dir}")

    print("[sft] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())