#importing the libraries
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./sample_NLP_model")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Prediction function
def predict(text_input):
    inputs = tokenizer(text_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    return prediction+1

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Review Analyser"
)

# Launch the interface
iface.launch()
