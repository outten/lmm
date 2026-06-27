# Training on a cloud GPU

The SFT pipeline (training/scripts/sft_train.py) needs an NVIDIA GPU
with CUDA. The README scripts use Unsloth + QLoRA so a single 24GB
card (RTX 4090) is enough for a 7B base; 40GB+ is comfortable for
14B; 80GB is required for full FT or 70B-class models.

This document recommends a provider based on our actual workload
(Qwen2.5-Math-7B-Instruct, ~50K SFT examples, ~4096 context, 2-3
epochs) and lays out the tradeoffs.

## TL;DR

  - Reliability matters more than price for a first training run.
  - Pick **Lambda Labs** for v0.1 — predictable, no spot interruption,
    $60-$130 for the whole run.
  - Switch to **RunPod** or **Vast.ai** once the recipe is stable and
    you want to iterate cheaply.

## What the run looks like

For v0.1 we expect ~30 hours of GPU time across:

  - Data conversion + sanity check (~1h, any GPU)
  - SFT (Phase 1) on ~50K examples, 2-3 epochs (~10-20h on A100/H100)
  - DPO (Phase 2) if you do it (~5-10h)
  - Eval (~2-4h)
  - GGUF conversion + Ollama build (~1h)

30 hours is the planning number for budgeting.

## Provider comparison (live prices, $/hr)

| GPU                | VRAM | Lambda | RunPod (Secure) | Vast.ai (median) | Vast.ai (low) |
|--------------------|------|--------|-----------------|------------------|---------------|
| H100 SXM           | 80GB | 4.29   | 2.89            | 2.00             | 1.47          |
| A100 SXM           | 80GB | 1.99   | 1.49            | 0.77             | 0.13          |
| A100 PCIe          | 40GB | 1.99   | 1.39            | 0.53             | 0.40          |
| RTX 6000 Ada       | 48GB | —      | 0.77            | ~0.50            | —             |
| RTX 4090           | 24GB | —      | 0.46            | 0.35             | 0.13          |

(Vast.ai is market-priced; rates change continuously. The "low" column
is the cheapest currently-available instance — usually interruptible.)

## Estimated v0.1 cost (~30 GPU-hours)

| Configuration                       | $/hr  | Total  | Notes                              |
|-------------------------------------|-------|--------|------------------------------------|
| Lambda 1x H100 SXM 80GB             | 4.29  | $129   | Reliable, default                  |
| Lambda 1x A100 SXM 80GB             | 1.99  | $60    | Same reliability, ~2x slower       |
| RunPod 1x H100 SXM 80GB             | 2.89  | $87    | Good managed mid-tier              |
| RunPod 1x A100 SXM 80GB             | 1.49  | $45    | Best managed value                 |
| Vast.ai H100 SXM (median)           | 2.00  | $60    | Risk of interruption               |
| Vast.ai A100 SXM4 80GB (median)     | 0.77  | $23    | Very cheap, slower + interruptible |
| Vast.ai RTX 4090 (low)              | 0.35  | $11    | Tight on VRAM, may OOM at seq 4096 |

## Recommendation per phase

### v0.1 (Foundation)

  Use **Lambda Labs 1x A100 SXM 80GB** or **RunPod 1x A100 SXM 80GB**.

  Reasoning:
    - 80GB VRAM is enough headroom for QLoRA on 7B at seq 4096 with
      reasonable batch size.
    - No spot interruption = no rebuilding state 8 hours in.
    - ~$60 for the whole run, which is small enough not to matter.
    - Both have one-line SSH/API provisioning.

  Concretely:

      # Lambda
      lambda cluster create --instance-type gpu_8x_a100_80gb_sxm4 \
          --name lmm-v01 --file-spec ../lambda-image

      # Or RunPod
      runpodctl create pod --name lmm-v01 \
          --image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel \
          --gpu-type "NVIDIA A100 80GB SXM" --container-disk-in-gb 200

### v0.2 (Code execution at scale)

  Reuse the same machine. Same dataset, longer training, more eval.
  Budget ~$100.

### v0.3 (Process supervision, larger base)

  Move to **H100** or **H200**. Qwen2.5-14B / 72B QLoRA needs faster
  interconnect; H100's 3.35 TB/s NVLink matters here.

  Recommended: Lambda 1x H100 SXM ($4.29/hr, ~$130 for the run) OR
  Vast.ai H100 SXM ($2.00/hr, ~$60 if you can survive an interruption).

  Risk note: H100 SXM spot is often interrupted on Vast. Schedule
  checkpoints every 30 minutes (the SFTTrainer already does this) and
  pick "interruptible=false" if available.

### v1.0 (Research-grade)

  Multi-node H100 cluster — out of scope for this doc. Talk to Lambda
  1-Click Clusters, CoreWeave, or AWS P5.

## What to actually do today

If you have $50-$150 budget and want to ship v0.1:

  1. Sign up at lambda.ai (10 minutes), add a card, request quota
     (often instant for A100 80GB, sometimes a wait for H100).
  2. SSH in, install our training deps:
         pip install -U unsloth transformers trl peft accelerate \
             datasets bitsandbytes
  3. Clone the repo, format a small dataset:
         python3 training/scripts/reformat_dataset.py \
             /path/to/metamath.jsonl -o training/data/sft.jsonl \
             --format metamath --limit 50000
  4. Train:
         python3 training/scripts/sft_train.py \
             --base Qwen/Qwen2.5-Math-7B-Instruct \
             --data training/data/sft.jsonl \
             --output training/runs/v0.1 \
             --epochs 2 --batch-size 4
  5. Convert + push:
         python3 training/scripts/convert_to_gguf.py training/runs/v0.1/merged \
             --out training/runs/v0.1/gguf --quant Q4_K_M
         ollama create outten/lmm:v0.1 -f Modelfile
         ollama push outten/lmm:v0.1

If you want to spend less and are willing to babysit:

  1. Vast.ai. Filter by "80GB VRAM" and "Data center" hosts (not
     residential). Sort by price.
  2. Use the `vastai` CLI to spin up. Image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`.
  3. Same training steps as above.
  4. Accept that the instance may be interrupted — Unsloth's checkpoint
     resumption will pick up where it left off.

## Free / cheap alternatives (lower priority)

  - **Colab Pro** ($10/mo): 1x A100 40GB. Enough for QLoRA 7B at
    shorter sequence lengths. Sessions cap at ~24h.
  - **Kaggle** (free): 1x Tesla T4 16GB — too small for 7B QLoRA
    without aggressive 4-bit offloading.
  - **Lightning AI studios** ($5 credit): spot H100s, can be cheap.
  - **Modal** (per-second): great DX but tends to be pricier for
    long-running training.

## What we still need to decide

  - Storage of training data and checkpoints: per-provider (S3,
    RunPod Network Volume, Vast instance disk). All cost extra.
  - Whether we want public, reproducible training (all on
    GitHub Actions-style runners) vs ad-hoc runs on a single
    cloud machine.

## References

  - Lambda Labs: https://lambda.ai/instances
  - RunPod:      https://www.runpod.io/pricing
  - Vast.ai:     https://vast.ai/pricing
  - Modal:       https://modal.com/pricing
  - CoreWeave:   https://www.coreweave.com/pricing
  - AWS P5:      https://aws.amazon.com/ec2/instance-types/p5/