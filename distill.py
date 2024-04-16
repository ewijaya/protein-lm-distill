import gc
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
import torch
import logging
from datasets import load_dataset
from torch.nn import functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming CUDA is available, set device to cuda, otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the teacher model and ensure it's on the right device
teacher_model_path = "nferruz/ProtGPT2"
teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_path).to(device)
teacher_tokenizer = GPT2Tokenizer.from_pretrained(teacher_model_path)

# Check and set the padding token for tokenizer if not already set
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Define the smaller student model configuration and ensure it's also on the same device
student_config = GPT2Config(
    vocab_size=teacher_model.config.vocab_size,
    n_positions=teacher_model.config.n_positions,
    n_ctx=teacher_model.config.n_ctx,
    n_embd=256,  # smaller size
    n_layer=4,  # fewer layers
    n_head=4,
    activation_function="gelu_new",
)
student_model = GPT2LMHeadModel(student_config).to(device)

# Load the dataset
data_files = {"train": "data/train_small.txt", "validation": "data/validate_small.txt"}
datasets = load_dataset("text", data_files=data_files)


# Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = teacher_tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=1024
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Define custom parameters for distillation
temperature = 2.0
alpha = 0.5


# Define a custom Trainer class to implement distillation
class DistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # Move inputs to the correct device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = inputs.get("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Get teacher model's logits with no_grad
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Compute the distillation loss
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        soft_loss = loss_fct(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
        )

        hard_loss = torch.nn.CrossEntropyLoss()(
            student_logits.view(-1, student_logits.size(-1)), labels.view(-1)
        )
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss

        return (loss, outputs) if return_outputs else loss


# Setup training arguments
training_args = TrainingArguments(
    output_dir="./models/protgpt2-distilled-test",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
    evaluation_strategy="epoch",
    save_strategy="no",
    save_total_limit=1,
)

# Initialize the trainer with the custom class
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=teacher_tokenizer,
)

# Start training
trainer.train()

# Manually save the model and tokenizer to the specified directory after training
model_save_path = training_args.output_dir
student_model.save_pretrained(model_save_path)
teacher_tokenizer.save_pretrained(model_save_path)

torch.cuda.empty_cache()
gc.collect()
