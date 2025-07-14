import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2", tokenizer="gpt2")

generator = load_generator()

def generate_cover_letter(resume_text, job_description):
    """
    Generate a concise, professional cover letter from resume and JD.
    """
    if not resume_text or not job_description:
        return "Please provide both resume and job description to generate a cover letter."

    resume_text = resume_text[:1000]
    job_description = job_description[:1000]

    prompt = f"""
Write a concise and professional cover letter tailored to the following job.

Candidate Resume:
{resume_text}

Job Description:
{job_description}

Cover Letter:
"""

    try:
        output = generator(prompt, max_length=400, num_return_sequences=1, do_sample=True, temperature=0.7)
        letter = output[0]["generated_text"].replace(prompt, "").strip()
        return letter
    except Exception as e:
        return f"❌ Error generating cover letter: {e}"
