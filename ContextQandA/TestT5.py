import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load trained model and tokenizer
model_path = "./t5_qa_model"  # Ensure this path is correct
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

def answer_question(question, context):
    input_text = f"question: {question} context: {context}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move to correct device

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    question = input("Enter your question: ")
    context = input("Enter the context (text from PDF or other source): ")
    
    answer = answer_question(question, context)
    print("\nPredicted Answer:", answer)
