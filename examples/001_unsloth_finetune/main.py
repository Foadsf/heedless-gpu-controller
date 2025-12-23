import os
import sys
import subprocess
import shutil
import torch

# --- 0. CLEANUP (The Fix for "NameError: psutil") ---
# We must wipe the Unsloth cache to force it to recognize the new dependencies.
if os.path.exists("unsloth_compiled_cache"):
    print("🧹 Wiping stale Unsloth cache...")
    shutil.rmtree("unsloth_compiled_cache")

# --- 1. HARDWARE CHECK ---
if torch.cuda.is_available():
    major_version = torch.cuda.get_device_capability()[0]
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 GPU Detected: {gpu_name} (Compute Capability {major_version}.x)")
    
    if major_version < 7:
        print("❌ ERROR: You are on a P100. Unsloth requires T4 or newer.")
        sys.exit(1)
else:
    print("❌ No GPU detected!")
    sys.exit(1)

# --- 2. SELF-CORRECTING SETUP ---
try:
    import unsloth
    import psutil
    print("✅ Dependencies detected.")
except ImportError:
    print("⚙️ Installing Dependencies (This takes ~5 mins)...")
    commands = [
        # 1. Install PyTorch 2.4.0 (Compatible with Kaggle Python 3.12)
        "pip install --upgrade --force-reinstall --no-cache-dir torch==2.4.0 triton --index-url https://download.pytorch.org/whl/cu121",
        # 2. Install Unsloth
        "pip install 'unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git'",
        # 3. Install Helpers (psutil is critical here)
        "pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes psutil",
    ]
    for cmd in commands:
        print(f"Executing: {cmd}")
        subprocess.check_call(cmd, shell=True)
    
    print("🔄 Restarting kernel to apply changes...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# --- 3. MAIN LOGIC ---
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Configuration
max_seq_length = 2048
dtype = None 
load_in_4bit = True 

# Data
raw_data = [
    {"prompt": "User:\nGenerate a profile for Mike.\n\nAssistant:\n", "response": "{\"name\": \"Mike\", \"job\": \"Coder\"}"},
    {"prompt": "User:\nWho is Igor?\n\nAssistant:\n", "response": "{\"name\": \"Igor\", \"job\": \"Guide\"}"},
    {"prompt": "User:\nDescribe Sarah.\n\nAssistant:\n", "response": "{\"name\": \"Sarah\", \"job\": \"Data Sci\"}"}
]
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(lambda x: {"text": x["prompt"] + x["response"] + "<|endoftext|>"})

# Model
print("🧠 Loading Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Fix for Device-side assert
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Train
print("🚂 Training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
trainer.train()

print("💾 Saving GGUF...")
try:
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    print("✅ DONE! Model saved successfully.")
except Exception as e:
    print(f"Error saving: {e}")
