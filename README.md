# generative-ai-projects
Projects on LLMs, Chatbots, and Computer Vision.
# Medbot - Personal Remedy Doctor Chatbot

Medbot is a personal remedy doctor chatbot built to provide information and answers based on uploaded medical documents in PDF format. It uses advanced NLP models such as Llama-2 and embeddings to analyze the content and provide relevant responses to user queries.

## Features

- Upload and process medical PDF documents
- Vectorize and store medical text data using embeddings
- Generate answers using Llama models based on the processed text
- Interactive chatbot interface using Streamlit

## Technologies Used

- **Streamlit**: For building the web interface
- **Llama-2**: Llama model used for generating answers
- **Hugging Face Transformers**: For loading and using pre-trained models
- **LangChain**: For handling LLM interactions and integrating with vector stores
- **PyPDF2**: For extracting text from uploaded PDF documents
- **FAISS**: For storing and retrieving vectorized text chunks

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/Medbot.git
   cd Medbot
