import os
import shutil
import gc
import json
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

# Load .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Advanced CUDA optimization
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
    print(f"🔧 Using dtype: {torch_dtype}")

    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("🚀 Using CPU")

# Use a larger model for advanced GPU training
MODEL_NAME = "microsoft/DialoGPT-medium"  # Medium model for better performance
print(f"📦 Model: {MODEL_NAME}")

print("📊 Loading dataset...")
# Even larger dataset for advanced training
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:100000]")

# More sophisticated filtering
dataset = dataset.filter(
    lambda x: len(x["text"].strip()) > 200 and len(x["text"].strip()) < 2000
)

try:
    dataset_size = len(dataset)  # type: ignore
    print(f"Dataset size after filtering: {dataset_size}")
except (TypeError, AttributeError, NotImplementedError):
    dataset_size = None
    print("Dataset loaded (size unknown)")

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
)

# Move model to GPU if available
if device == "cuda":
    model = model.to(device)  # type: ignore
    torch.cuda.empty_cache()

# Advanced model optimizations
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Optional: Apply LoRA or other parameter-efficient fine-tuning
# Uncomment the following lines if you want to use LoRA
# from peft import LoraConfig, get_peft_model, TaskType
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
# )
# model = get_peft_model(model, lora_config)

print("✂️  Tokenizing dataset...")


def tokenize(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,  # Longer context for advanced training
        padding=False,
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


# Process in large batches for GPU efficiency
tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    batch_size=2000,  # Very large batches for GPU
    remove_columns=["text"],
)

# Set format for PyTorch
if hasattr(tokenized_dataset, "set_format"):
    tokenized_dataset.set_format(  # type: ignore
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

# Clear memory before training
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

print("⚙️  Setting up advanced training...")
training_args = TrainingArguments(
    output_dir="./gpu-advanced-finetuned",
    # Advanced GPU batch settings
    per_device_train_batch_size=16,  # Large batch size for GPU
    gradient_accumulation_steps=1,  # Effective batch size of 16
    # Training parameters
    num_train_epochs=5,  # More epochs for better convergence
    learning_rate=3e-5,  # Lower learning rate for stability
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=0.1,  # Warmup for better training stability
    # Advanced GPU settings
    fp16=torch_dtype == torch.float16,
    bf16=torch_dtype == torch.bfloat16,
    dataloader_pin_memory=True,
    dataloader_num_workers=8,  # More workers for faster data loading
    # Advanced optimizations
    dataloader_prefetch_factor=4,
    dataloader_persistent_workers=True,
    # Logging and saving
    logging_steps=25,
    save_steps=250,
    save_total_limit=5,
    eval_steps=500,
    evaluation_strategy="steps",
    # Memory optimization
    remove_unused_columns=True,
    dataloader_drop_last=True,
    save_safetensors=True,
    # Performance optimizations
    dataloader_disable_tqdm=False,
    # Disable unnecessary features
    report_to=None,
    push_to_hub=False,
    # Advanced settings
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_only_model=True,
)

# Advanced data collator with dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

# Create a small validation set
if dataset_size and dataset_size > 1000:
    train_size = int(0.9 * dataset_size)
    eval_dataset = tokenized_dataset.select(range(train_size, dataset_size))
    train_dataset = tokenized_dataset.select(range(train_size))
else:
    train_dataset = tokenized_dataset
    eval_dataset = None

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("🏃 Starting advanced training...")
try:
    # Clear memory before training
    if device == "cuda":
        torch.cuda.empty_cache()

    # Training with progress tracking
    trainer.train()
    print("✅ Advanced training completed successfully!")

except Exception as e:
    print(f"❌ Training failed with error: {e}")
    print("💾 Attempting to save partial model...")

print("💾 Saving model...")
try:
    # Clean up existing directory
    output_dir = "./gpu-advanced-finetuned"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")

    # Save detailed training info
    training_info = {
        "base_model": MODEL_NAME,
        "dataset_size": dataset_size if dataset_size is not None else "unknown",
        "device_used": device,
        "max_sequence_length": 1024,
        "batch_size": "16 x 1 (gradient accumulation)",
        "epochs": 5,
        "mixed_precision": str(torch_dtype),
        "gpu_optimizations": "enabled",
        "flash_attention": torch.cuda.is_available(),
        "lora": False,  # Set to True if LoRA was used
        "warmup_ratio": 0.1,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
    }

    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    # Also save as text for easy reading
    with open(f"{output_dir}/training_info.txt", "w") as f:
        for key, value in training_info.items():
            f.write(f"{key}: {value}\n")

    print("📋 Detailed training info saved")

except Exception as e:
    print(f"❌ Failed to save model: {e}")
    print("🔍 Check disk space and permissions")

# Final cleanup
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

print("🎉 Advanced GPU training script completed!")
