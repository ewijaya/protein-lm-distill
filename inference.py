from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
import re

# Load the model and tokenizer
# model_name = "nferruz/protgpt2"
model_name = "models/protgpt2-distilled-t2.0-a0.5-l4-h4-e256"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure tokenizer is padding from the left
tokenizer.padding_side = "left"

# Initialize the pipeline
text_generator = TextGenerationPipeline(
    model=model, tokenizer=tokenizer, device=0
)  # specify device if needed

# Generate sequences
sequences = text_generator(
    "<|endoftext|>KKAW",
    max_length=100,
    do_sample=True,
    top_k=950,
    repetition_penalty=1.2,
    num_return_sequences=10,
    pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
    eos_token_id=0,
    truncation=True,
)

for i, seq in enumerate(sequences):
    # Remove "<|endoftext|>"
    seq["generated_text"] = seq["generated_text"].replace("<|endoftext|>", "")

    # Remove newline characters and non-alphabetical characters
    seq["generated_text"] = "".join(
        char for char in seq["generated_text"] if char.isalpha()
    )
    print(f">Seq_{i}")
    print(seq["generated_text"])
