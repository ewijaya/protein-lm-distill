from transformers import pipeline

# model_name = "nferruz/ProtGPT2"
model_name = "models/protgpt2-distilled-test/checkpoint-300"
protgpt2 = pipeline("text-generation", model=model_name)
sequences = protgpt2(
    "<|endoftext|>",
    max_length=100,
    do_sample=True,
    top_k=950,
    repetition_penalty=1.2,
    num_return_sequences=10,
    eos_token_id=0,
)
for seq in sequences:
    print(seq)
