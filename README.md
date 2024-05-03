## Using Langchain chains and Memory

This is a simple project using models such as Zephyr and Mistral 8x7B-it model from HuggingFace to do chaining using Langchain with memory.

### RAG using Langchain to build chatbot
#### Steps:
1) Import libraries
2) Initialize embedding model to generate emebddings for user queries and documents
3) Create document and split document into chunks
4) Create embeddings for chunks and store them in FAISS vector store
5) Get user input and convert the input to embedding
6) Use embedding from user input to search most relevant document chunk in vector store.
7) Display the final result
