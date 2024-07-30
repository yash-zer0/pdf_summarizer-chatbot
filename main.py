import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
from huggingface_hub import login

# Use your token here
login(token="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")


# Initialize Hugging Face models
summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
chat_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

# Set up the Hugging Face token







# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit app layout
st.title("PDF Summarization Chatbot")
st.write("Upload a PDF to summarize its contents:")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted Text:")
    st.write(text)

    # Summarize text
    if st.button("Summarize"):
        summary = summary_pipeline(text, max_length=500, min_length=100, do_sample=False)
        st.write("Summary:")
        st.write(summary[0]['summary_text'])

    # Chatbot interface
    user_input = st.text_input("Chat with the bot:")
    if user_input:
        response = chat_pipeline(user_input, max_length=150, num_return_sequences=1)
        st.write("Chatbot Response:")
        st.write(response[0]['generated_text'])
