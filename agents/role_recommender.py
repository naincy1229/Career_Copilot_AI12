# agents/role_recommender.py

from transformers import pipeline

def suggest_roles(resume_data, model="distilgpt2"):
    if not resume_data:
        return "Resume data is missing. Please upload a valid resume."

    prompt = (
        "You are a career advisor. Based on the following resume data:\n\n"
        f"{resume_data}\n\n"
        "Suggest the top 3 most suitable job roles for this candidate and explain why."
    )

    try:
        generator = pipeline("text-generation", model=model)
        output = generator(prompt, max_length=250, do_sample=True, temperature=0.7)
        return output[0]["generated_text"].replace(prompt, "").strip()

    except Exception as e:
        return f"❌ Error generating role recommendations: {str(e)}"
