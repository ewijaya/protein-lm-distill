#!/home/ubuntu/storage1/anaconda3/envs/pepmlm/bin/python
import gc
import os
import shutil
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)
import torch
import logging
from datasets import load_dataset, DatasetDict
from torch.nn import functional as F
import json
import wandb
import glob
import argparse

# How to run:
# $ conda activate pepmlm
# $ nohup sh -c './distill_using_nferruz_dataset.py --temperature 2.0 --alpha 0.5 > nohup.out && stopinstance' &
## with && we ensure stopinstance only run if distill.py runs successfully

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Distillation parameters")
parser.add_argument(
    "--temperature", type=float, default=2.0, help="Temperature for distillation"
)
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for distillation")
parser.add_argument("--n_embd", type=int, default=256, help="Embedding size")
parser.add_argument("--n_layer", type=int, default=4, help="Number of layers")
parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
parser.add_argument(
    "--train_size_prop",
    type=float,
    default=0.1,
    help="Proportion of the validation set to use as the training set",
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
)
args = parser.parse_args()


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the teacher model
teacher_model_path = "nferruz/ProtGPT2"
teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_path).to(device)

# Setup tokenizer
teacher_tokenizer = GPT2TokenizerFast.from_pretrained(teacher_model_path)

if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_tokenizer.padding_side = "left"


# Define the student model configuration
def create_student_config(n_embd, n_layer, n_head):
    return GPT2Config(
        vocab_size=teacher_model.config.vocab_size,
        n_positions=teacher_model.config.n_positions,
        n_ctx=teacher_model.config.n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        activation_function="gelu_new",
        bos_token_id=teacher_model.config.bos_token_id,
        eos_token_id=teacher_model.config.eos_token_id,
    )


# Load the dataset from parquet files
local_dataset_path = "/home/ubuntu/storage2/various_hugging_face_data_and_models/data"
data_files = {"train": glob.glob(f"{local_dataset_path}/train*.parquet")}

# Load the training set
dataset = load_dataset("parquet", data_files=data_files, trust_remote_code=True)

# Subsample the dataset: prop=1 -> 100%, prop=0.1 -> 10%
train_subset = dataset["train"].train_test_split(
    train_size=args.train_size_prop
)["train"]
print(f"Subset size: {len(train_subset)}")

# Use the subset as the training dataset
dataset = DatasetDict({"train": train_subset})


# The parquet files are already tokenized with input_ids, attention_mask, and labels
# We can use the dataset directly without additional tokenization
tokenized_dataset = dataset
print(f"Dataset loaded with {len(tokenized_dataset['train'])} samples")


# Define a custom Trainer class for distillation
class DistillationTrainer(Trainer):
    def __init__(self, temperature, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = teacher_model
        self.training_logs = []  # Initialize an empty list to store training logs

    def log(self, logs: dict):
        super().log(logs)
        # Store the log data in the list
        self.training_logs.append(logs)

    def save_logs(self, save_path):
        # Save the collected logs to a JSON file
        with open(save_path, "w") as f:
            json.dump(self.training_logs, f, indent=4)

    def compute_loss(self, model, inputs, return_outputs=False):
        r"""
        Compute the combined knowledge distillation loss for a student model during training.

        This method calculates the distillation loss by combining two types of losses:
        1. Soft Loss (Knowledge Distillation Loss): Helps the student model learn the
        generalized output distribution of the teacher model using softened probabilities.
        2. Hard Loss (Cross-Entropy Loss): Encourages the student model to predict the
        correct classes based on ground truth labels, ensuring practical prediction capabilities.

        The combined loss is a weighted sum of these two losses, controlled by a hyperparameter
        alpha (self.alpha), allowing customization based on specific needs.

        Parameters:
            model (torch.nn.Module): The student model being trained.
            inputs (dict): A dictionary containing input tensors needed for model forward pass.
                        Expected to include keys like 'input_ids' and 'labels' for BERT-like models.
            return_outputs (bool, optional): Flag to determine if model outputs should be returned
                                            with the loss. Defaults to False.

        Returns:
            tuple or torch.Tensor: If return_outputs is True, returns a tuple of (loss, outputs),
                                otherwise returns the loss tensor directly.

        Soft Loss Formula:
            L_{soft} = \text{KL}(\text{softmax}(\frac{s}{T}), \text{softmax}(\frac{t}{T}))
        where s are the logits from the student model, t are the logits from the teacher model,
        T is the temperature, and KL is the Kullback-Leibler divergence.

        Hard Loss Formula:
            L_{hard} = -\sum_{i} y_i \log(\text{softmax}(s_i))
        where y_i is the true label and s_i are the logits from the student model.

        Combined Loss Formula:
            L = \alpha L_{hard} + (1 - \alpha) L_{soft}
        where \alpha is the weight factor balancing the two losses.

        The method ensures all operations are performed on the same device as the student model.

        References:
            Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.
            arXiv:1503.02531. This paper introduces knowledge distillation, where the soft targets
            generated by the teacher model provide additional training signals to the student model.
        """

        # Make it more robust so that it can run with multiple GPUs.
        if isinstance(model, torch.nn.DataParallel):
            device = model.module.device
        else:
            device = model.device

        inputs = {
            k: v.to(device) for k, v in inputs.items()
        }  # Ensure all inputs are on the same device
        labels = inputs.get("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Move teacher model to the same device and compute its outputs
        self.teacher_model = self.teacher_model.to(device)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Shift logits and labels for causal LM (predict next token)
        # logits[i] predicts token[i+1], so we align them properly
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate the soft loss using KL divergence
        # Scale by T^2 as per Hinton et al. (2015) to maintain gradient magnitude
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        soft_loss = loss_fct(
            F.log_softmax(shift_student_logits / self.temperature, dim=-1),
            F.softmax(shift_teacher_logits / self.temperature, dim=-1),
        )

        # Calculate the hard loss using cross-entropy on shifted sequences
        hard_loss = torch.nn.CrossEntropyLoss()(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Combine the soft and hard losses
        # T^2 scaling for soft loss maintains proper gradient contribution
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * (self.temperature ** 2) * soft_loss

        return (loss, outputs) if return_outputs else loss


# Create the student model configuration based on the command-line arguments
student_config = create_student_config(args.n_embd, args.n_layer, args.n_head)
student_model = GPT2LMHeadModel(student_config).to(device)

# Create the model name based on temperature, alpha, and architecture parameters
model_name = f"protgpt2-distilled-t{args.temperature}-a{args.alpha}-l{args.n_layer}-h{args.n_head}-e{args.n_embd}-p{args.train_size_prop}-lr{args.learning_rate:.0e}.uniprot"

# Initialize Weights & Biases with the model name as the run name
wandb.init(
    project="PROTGPT2_DISTILLATION",
    name=model_name,
    settings=wandb.Settings(_service_wait=120),
)

output_dir = f"./models/{model_name}"

if os.path.exists(output_dir):
    print(f"Output directory {output_dir} already exists. Deleting...")
    shutil.rmtree(
        output_dir
    )  # Use shutil.rmtree to delete the directory and its contents
else:
    print(f"Output directory {output_dir} does not exist. Creating...")

# Setup training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    logging_strategy="steps",  # Log at the end of each steps for prettier plot.
    logging_steps=10,
    save_strategy="no",
    save_total_limit=1,
    fp16=True,
)

# Initialize the trainer
trainer = DistillationTrainer(
    temperature=args.temperature,
    alpha=args.alpha,
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=teacher_tokenizer,
)

# Start training
trainer.train()

# Save the model and tokenizer
model_save_path = training_args.output_dir
trainer.save_logs(f"{model_save_path}/training_logs.json")
student_model.save_pretrained(model_save_path)
teacher_tokenizer.save_pretrained(model_save_path)

# Serialize training arguments and distillation parameters
hyperparams = {
    "training_arguments": training_args.to_dict(),
    "distillation_temperature": trainer.temperature,
    "distillation_alpha": trainer.alpha,
    "model_architecture": {
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
    },
}
with open(f"{model_save_path}/training_hyperparameters.json", "w") as f:
    json.dump(hyperparams, f, indent=4)

wandb.finish()  # Ensure this is before clearing GPU memory

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()
