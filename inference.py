from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
import re

# Load the model and tokenizer
model_name = "models/protgpt2-distilled-test/"
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
    "",
    max_length=30,
    do_sample=True,
    top_k=950,
    repetition_penalty=1.2,
    num_return_sequences=10,
    pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
    eos_token_id=0,
    truncation=True,
)

for i, seq in enumerate(sequences):
    text = seq["generated_text"]
    cleaned_text = re.sub("[^a-zA-Z]", "", text)
    print(f">Seq_{i}")
    print(cleaned_text)
