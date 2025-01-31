from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "llama2-7b-llama_cpp-ggmlv3-q4_1"  # Example model name, adjust as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
