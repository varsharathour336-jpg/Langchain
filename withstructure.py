from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="google/flan-t5-base"
)

text = """
The Alchemist by Paulo Coelho is a deeply inspiring story about a young shepherd
named Santiago who dreams of finding treasure. His journey teaches him about destiny,
faith, and listening to his heart.
"""

summary = summarizer(
    text,
    max_length=60,
    min_length=20,
    do_sample=False
)

print(summary)