# chains/interview_agent.py

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def run_mock_interview(question):
    try:
        # Encode input
        inputs = tokenizer.encode(question + tokenizer.eos_token, return_tensors="pt")

        # Generate response
        outputs = model.generate(
            inputs,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and return
        reply = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return reply

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"
