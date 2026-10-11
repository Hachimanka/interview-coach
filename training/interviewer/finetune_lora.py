"""G4: LoRA fine-tune of the interviewer model on the synthetic data (Google Colab, free T4 GPU is enough).

    !pip install -r requirements-common.txt transformers peft trl datasets accelerate
    !python -m training.interviewer.finetune_lora --base Qwen/Qwen2.5-1.5B-Instruct \
        --data data/synthetic/interviewer.jsonl --out models/interviewer-1.5b

Then convert to GGUF Q4_K_M for llama.cpp (same format as the base model):
    !git clone https://github.com/ggerganov/llama.cpp && pip install -r llama.cpp/requirements.txt
    !python llama.cpp/convert_hf_to_gguf.py models/interviewer-1.5b/merged --outfile interviewer-1.5b-f16.gguf
    !cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build --target llama-quantize -j
    !llama.cpp/build/bin/llama-quantize interviewer-1.5b-f16.gguf interviewer-1.5b-q4_k_m.gguf Q4_K_M

Copy interviewer-1.5b-q4_k_m.gguf to models/llm/ on the PC, run eval/interviewer_eval.py with
--model interviewer-1.5b-q4 and the base model, and switch reply.llm_model only if it is better.
Repeat with --base Qwen/Qwen2.5-0.5B-Instruct for the Pi (interviewer-0.5b-q4).
"""
from __future__ import annotations

import argparse
import random

from training.interviewer.dataset import read_jsonl, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/synthetic/interviewer.jsonl")
    ap.add_argument("--out", default="models/interviewer-1.5b")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    records = [r for r in read_jsonl(args.data) if validate(r)[0]]
    random.Random(0).shuffle(records)
    n_val = max(20, len(records) // 20)
    train, val = records[n_val:], records[:n_val]
    print(f"train {len(train)}  val {len(val)}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="auto")
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8, logging_steps=20,
                    eval_strategy="epoch", save_strategy="epoch", fp16=True, max_length=3072,
                    assistant_only_loss=True, report_to=[])
    trainer = SFTTrainer(model=model, args=cfg, peft_config=lora, processing_class=tok,
                         train_dataset=Dataset.from_list([{"messages": r["messages"]} for r in train]),
                         eval_dataset=Dataset.from_list([{"messages": r["messages"]} for r in val]))
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{args.out}/merged")
    tok.save_pretrained(f"{args.out}/merged")
    print(f"merged model saved to {args.out}/merged; convert it to GGUF (see the top of this file)")


if __name__ == "__main__":
    main()
