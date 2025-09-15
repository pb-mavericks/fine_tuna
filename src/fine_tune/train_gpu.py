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

# CUDA optimization: Check device availability
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16  # Use fp16 for better GPU performance
    print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("🚀 Using CPU")

# Use Phi-3 model for better performance
MODEL_NAME = (
    "microsoft/Phi-3-mini-4k-instruct"  # Latest Phi-3 model with instruction tuning
)
print(f"📦 Model: {MODEL_NAME}")

print("📊 Loading dataset...")
# Larger dataset size for GPU training
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:50000]")

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

print("🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
    torch_dtype=torch_dtype,
)

# Move model to GPU if available
if device == "cuda":
    model = model.to(device)  # type: ignore
    # Clear CUDA cache
    torch.cuda.empty_cache()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Important for training

print("✂️  Tokenizing dataset...")


def tokenize(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # Reasonable context length for GPU training
        padding=False,
        return_tensors="pt",
    )
    # Add labels directly in tokenization
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


# Process in larger batches for GPU efficiency
tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    batch_size=1000,  # Larger batches for GPU
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

print("⚙️  Setting up training...")
training_args = TrainingArguments(
    output_dir="./phi3-finetuned",
    # GPU-optimized batch settings for Phi-3 (3.8B parameters)
    per_device_train_batch_size=2,  # Smaller batch size for Phi-3
    gradient_accumulation_steps=8,  # Effective batch size of 16
    # Training parameters
    num_train_epochs=3,
    learning_rate=1e-5,  # Conservative learning rate for Phi-3
    weight_decay=0.01,
    max_grad_norm=1.0,
    # GPU-specific settings
    fp16=True,  # Enable mixed precision for better GPU performance
    dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
    dataloader_num_workers=4,  # Use multiple workers for data loading
    # Logging and saving
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    # Memory optimization
    remove_unused_columns=True,
    dataloader_drop_last=True,
    save_safetensors=True,
    # Performance optimizations
    dataloader_prefetch_factor=2,
    # Disable unnecessary features
    report_to=None,
    push_to_hub=False,
    # GPU-specific optimizations
    dataloader_persistent_workers=True,
    eval_strategy="no",  # Skip evaluation for faster training
)

# Memory-efficient data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # Better memory alignment for GPU
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
    if device == "cuda":
        torch.cuda.empty_cache()

    trainer.train()
    print("✅ Training completed successfully!")

except Exception as e:
    print(f"❌ Training failed with error: {e}")
    print("💾 Attempting to save partial model...")

print("💾 Saving model...")
try:
    # Clean up existing directory
    output_dir = "./phi3-finetuned"
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
        f.write("Max sequence length: 512\n")
        f.write("Batch size: 2 x 8 (gradient accumulation)\n")
        f.write("Epochs: 3\n")
        f.write("Mixed precision: fp16\n")
        f.write("GPU optimizations: enabled\n")

    print("📋 Training info saved")

except Exception as e:
    print(f"❌ Failed to save model: {e}")
    print("🔍 Check disk space and permissions")

# Final cleanup
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

print("🎉 Script completed!")
