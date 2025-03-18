import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Model and Tokenizer
MODEL_PATH = "best_model_odd_student"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource  # Cache model for better performance
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # Start an async event loop if not running

model, tokenizer = load_model()


# Define labels
LABELS = ["Non-Hate", "Offensive", "Hate"]

# Streamlit UI
st.title("🛑 Hate Speech Detection Web App")
st.markdown("Enter a sentence, and the model will classify it as **Non-Hate, Offensive, or Hate Speech**.")

# User Input
user_input = st.text_area("Enter your text here:", "")

# Predict Button
if st.button("Classify Text"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Display result
        st.success(f"Predicted Category: **{LABELS[predicted_class]}**")