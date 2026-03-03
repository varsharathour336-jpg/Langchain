from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Create HuggingFace pipeline
hf_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100
)

# Wrap it for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Pass LLM into ChatHuggingFace
model = ChatHuggingFace(llm=llm)

while True:
  user_input=input('You:')
  if user_input=='exit':
    break
  result=model.invoke(user_input)
  print("AI:",result.content)