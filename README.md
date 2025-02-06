# Context-Based Question Answering (T5 Model)

This project implements a **context-aware Question Answering system** using the **T5 transformer model**. The system takes user input (context and question) and generates an answer using a fine-tuned T5 model trained on the **SQuAD** dataset.

## Features
- Train a T5 model on the **SQuAD** dataset (`TrainT5.py`)
- Test and generate answers using the trained model (`TestT5.py`)
- User-friendly Streamlit interface (`frontend.py`)
- Supports manual text input and context for question answering

---

## 1️⃣ Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

Install dependencies:
```sh
pip install fastapi uvicorn torch transformers datasets streamlit
```

---

## 2️⃣ Training the Model
To train the model using the **SQuAD** dataset, run:
```sh
python TrainT5.py
```
This script will:
- Load the **T5 model** (`t5-base` by default)
- Preprocess the SQuAD dataset
- Train the model for **3 epochs**
- Save the model to `./t5_qa_model`

---

## 3️⃣ Testing the Model
To manually test the model, run:
```sh
python TestT5.py
```
It will prompt you for a **question** and **context**, then generate an answer.

Example:
```
Enter your question: What is AI?
Enter the context: AI stands for Artificial Intelligence. It is used in many fields.

Predicted Answer: Artificial Intelligence
```

---

## 4️⃣ Running the Streamlit Web Interface
To launch the **frontend UI**, run:
```sh
streamlit run frontend.py
```
This will open a web app where you can:
- Enter a **question**
- Provide **optional context**
- Get an answer from the trained model

---
## 6️⃣ Troubleshooting
- If training runs out of memory, lower `batch_size` in `TrainT5.py`
- Ensure the model path in `TestT5.py` and `frontend.py` matches the saved model directory
- If Streamlit doesn't load, check for missing dependencies with `pip list`

---

## 7️⃣ Future Enhancements
- Add file upload support for long-form context processing
- Implement image-based OCR for extracting text from images
- Extend model to support multilingual question answering

## License
This project is licensed under the **MIT License**.

