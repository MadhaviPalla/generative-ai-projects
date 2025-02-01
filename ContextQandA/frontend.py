import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load trained model
def load_model(model_path="./saved_t5_model"):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Generate an answer using the model
def generate_answer(question, context, tokenizer, model):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Streamlit frontend
def main():
    st.title("SQuAD-based QA Chatbot")
    st.subheader("Ask any question based on the given context (optional).")
    
    # User inputs
    user_question = st.text_input("Enter your question:")
    user_context = st.text_area("Provide context (optional):", "")

    # Load model once
    tokenizer, model = load_model()

    # If submit button clicked
    if st.button("Get Answer"):
        if user_question:
            context = user_context if user_context else " "
            answer = generate_answer(user_question, context, tokenizer, model)
            st.write(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
