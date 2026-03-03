from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100
)

prompt = "<|user|>\nWhERE IS NITJ SITUATED?\n<|assistant|>\n"

output = pipe(prompt)

print(output[0]["generated_text"])
