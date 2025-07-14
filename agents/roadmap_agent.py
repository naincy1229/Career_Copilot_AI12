# utils/roadmap_agent.py

from transformers import pipeline

# Load model once using Streamlit cache
import streamlit as st

@st.cache_resource
def load_roadmap_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_roadmap_generator()

def generate_learning_roadmap(resume_data, query, model="gpt2"):
    if not query:
        return "❌ Please provide a valid learning goal or topic."

    # Prepare prompt
    prompt = f"""
You are an expert career coach. The user has the following background:

{resume_data}

They are asking for a learning roadmap on:
"{query}"

Generate a detailed, step-by-step personalized learning roadmap that includes beginner to advanced concepts, tools, and platforms.
"""

    try:
        output = generator(prompt, max_length=500, do_sample=True, num_return_sequences=1)
        roadmap = output[0]['generated_text'].replace(prompt, "").strip()
        return roadmap

    except Exception as e:
        return f"❌ Error generating roadmap: {e}"
