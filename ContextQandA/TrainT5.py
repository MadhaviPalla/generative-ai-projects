import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Pre-trained T5 Model and Tokenizer
model_name = "t5-base"  # Change to "t5-large" or "t5-11b" if needed
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Load SQuAD dataset
dataset = load_dataset("squad")

# Preprocessing function
def preprocess_data(example):
    input_text = f"question: {example['question']} context: {example['context']}"
    target_text = example["answers"]["text"][0] if example["answers"]["text"] else "No answer"
    
    model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(target_text, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
train_data = dataset["train"].map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)
valid_data = dataset["validation"].map(preprocess_data, batched=True, remove_columns=dataset["validation"].column_names)

# Set format for PyTorch
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
valid_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5_qa_model",
    eval_strategy="epoch",  # Updated from deprecated evaluation_strategy
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-5,
    per_device_train_batch_size=4,  # Reduce if running out of memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=2,  # Helps with memory efficiency
    fp16=True,  # Enables mixed precision training (if using GPU)
    push_to_hub=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data
)

# Start training
trainer.train()
