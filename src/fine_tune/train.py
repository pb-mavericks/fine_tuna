import os
import shutil
import gc
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

# MPS optimization: Check device availability
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32  # MPS works better with float32
    print("🚀 Using MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("🚀 Using CPU")

# Use a model optimized for MPS
MODEL_NAME = "microsoft/DialoGPT-small"  # Small, MPS-friendly
print(f"📦 Model: {MODEL_NAME}")

print("📊 Loading dataset...")
# Reasonable dataset size for MPS
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10000]")

# Filter out very short texts for better training quality
dataset = dataset.filter(lambda x: len(x["text"].strip()) > 100)

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

# Remove token limit for maximum context length

print("🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
)

# Move model to MPS if available
if device == "mps":
    model = model.to(device)  # type: ignore
    # Clear MPS cache
    torch.mps.empty_cache()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Important for training

print("✂️  Tokenizing dataset...")


def tokenize(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=False,  # Remove truncation to allow full context
        padding=False,  # Remove padding to avoid unnecessary tokens
        return_tensors="pt",
    )
    # Add labels directly in tokenization
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


# Process in smaller batches for MPS memory efficiency
tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    batch_size=100,  # Smaller batches for MPS
    remove_columns=["text"],
)

# Set format for PyTorch (only if it's a regular Dataset, not IterableDataset)
if hasattr(tokenized_dataset, "set_format"):
    tokenized_dataset.set_format(  # type: ignore
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

# Clear memory before training
gc.collect()
if device == "mps":
    torch.mps.empty_cache()

print("⚙️  Setting up training...")
training_args = TrainingArguments(
    output_dir="./mps-finetuned",
    # MPS-optimized batch settings
    per_device_train_batch_size=2,  # Slightly larger for efficiency
    gradient_accumulation_steps=4,  # Effective batch size of 8
    # Training parameters
    num_train_epochs=3,  # More epochs for better learning
    learning_rate=3e-5,  # Slightly lower for stability
    weight_decay=0.01,
    max_grad_norm=1.0,  # Gradient clipping
    # MPS-specific settings
    fp16=False,  # MPS doesn't fully support fp16
    dataloader_pin_memory=False,  # MPS doesn't support pinned memory
    dataloader_num_workers=0,  # Avoid multiprocessing issues on MPS
    # Logging and saving
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    # Memory optimization
    remove_unused_columns=True,
    dataloader_drop_last=True,
    save_safetensors=True,
    # Disable unnecessary features
    report_to=None,  # No wandb/tensorboard
    push_to_hub=False,
)

# Memory-efficient data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # Better memory alignment
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # type: ignore
    data_collator=data_collator,
)

print("🏃 Starting training...")
try:
    # Clear memory before training
    if device == "mps":
        torch.mps.empty_cache()

    trainer.train()
    print("✅ Training completed successfully!")

except Exception as e:
    print(f"❌ Training failed with error: {e}")
    print("💾 Attempting to save partial model...")

print("💾 Saving model...")
try:
    # Clean up existing directory
    output_dir = "./mps-finetuned"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")

    # Save training info
    with open(f"{output_dir}/training_info.txt", "w") as f:
        f.write(f"Base model: {MODEL_NAME}\n")
        f.write(
            f"Dataset size: {dataset_size if dataset_size is not None else 'unknown'}\n"
        )
        f.write(f"Device used: {device}\n")
        f.write("Max sequence length: 256\n")
        f.write("Batch size: 2 x 4 (gradient accumulation)\n")
        f.write("Epochs: 3\n")

    print("📋 Training info saved")

except Exception as e:
    print(f"❌ Failed to save model: {e}")
    print("🔍 Check disk space and permissions")

# Final cleanup
if device == "mps":
    torch.mps.empty_cache()
gc.collect()

print("🎉 Script completed!")
