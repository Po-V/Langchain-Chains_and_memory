# QA Chatbot with Retrieval Augmented Generation (RAG)

This repository contains a Python implementation of a chatbot with retrieval-augmented generation, using GPT model from HuggingFace, LangChain, and Streamlit. 
The chatbot is capable of answering questions based on a provided corpus of text data.

## Features

- Retrieval-augmented generation using GPT model from HuggingFace and LangChain
- FAISS database to store and retrieve external data
- Streamlit web interface for user interaction
- Customizable CSS styles for an enhanced chat experience

## Installation

1. Clone the repository:

```bash
https://github.com/Po-V/Langchain-Chains_and_memory.git
```

2. Navigate to the project directory:

```bash
cd Langchain-Chains_and_memory
```

3. Create a new virtual environment (optional but recommended):

```bash
python -m venv env
```

4. Activate the virtual environment (on Windows):

```bash
env\Scripts\activate
```

5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run QA_Chatbot.py
```
3. The chatbot will be accessible at http://localhost:8501 in your web browser.
4. Enter your questions in the input field, and the chatbot will provide relevant responses based on the provided web page.

## Configuration

You can customize the chatbot's behavior by modifying the following variables in the `QA_Chatbot.py` file:

- `model_name`: Replace this string with the name of the Hugging Face model you want to use.
- `corpus`: Replace this string with the link of corpus you want to use for the chatbot's knowledge base.
- `chain_type`: Modify the `chain_type` parameter in the `RetrievalQA` instantiation if you want to use a different chain type.
- `chunk_size` and `chunk_overlap`: Adjust these parameters in the `CharacterTextSplitter` to control the size and overlap of text chunks during indexing.


