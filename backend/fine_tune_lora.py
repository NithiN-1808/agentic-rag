import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load base model and tokenizer
BASE_MODEL = "bert-base-uncased"
OUTPUT_DIR = r"C:\ME Projects\agentic_rag\backend" 

# Load tokenizer and model for sequence classification
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,                # Low-rank size
    lora_alpha=32,       # LoRA scaling factor
    lora_dropout=0.1,    # Dropout for LoRA layers
    bias="none",         # No bias in LoRA
    task_type="SEQ_CLS"  # Sequence classification
)

# Wrap the base model with LoRA
model = get_peft_model(model, lora_config)

# Load a dataset (replace with your data)
dataset = load_dataset("glue", "mrpc")

def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    push_to_hub=False
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned LoRA model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")
