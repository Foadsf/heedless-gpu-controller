# === Unsloth Phi-3 Mini Fine-Tuning + Merge + GGUF Export ===
# Uses STANDARD trl.SFTTrainer to bypass Unsloth's psutil-patched trainer

import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout.decode())

run_command('pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"')
run_command('pip install trl peft accelerate bitsandbytes transformers datasets')

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import json
from trl import SFTTrainer  # <-- STANDARD trainer, no psutil!
from transformers import TrainingArguments

print("=== Loading model ===")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("=== Adding LoRA adapters ===")
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# Test dataset
data = [{"prompt": "Example prompt here.", "response": {"key": "value"}}]  # Replace later
ds = Dataset.from_list(data)

def to_text(ex):
    resp = json.dumps(ex["response"], ensure_ascii=False)
    return {"text": tokenizer.apply_chat_template([{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": resp}], tokenize=False, add_generation_prompt=False)}

dataset = ds.map(to_text, remove_columns=ds.column_names)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer.train()

model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
model.save_pretrained_merged("phi3_finetuned_merged", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("phi3_finetuned_gguf", tokenizer, quantization_method="q4_k_m")

print("=== DONE! GGUF in phi3_finetuned_gguf/ ===")
