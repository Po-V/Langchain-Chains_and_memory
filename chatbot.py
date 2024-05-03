import os
from constants import hf_key
import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import AsyncChromiumLoader
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
# load hf key
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_key


# website to be scrapped
movie_wiki = [
    'https://en.wikipedia.org/wiki/Argylle',
    'https://en.wikipedia.org/wiki/Mean_Girls_(2024_film)'
    
]

# load data into streamlit session state
if 'vector' not in st.session_state:
    # get embedding model
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    # load the website
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    # load the webiste into documents
    st.session_state.docs=st.session_state.loader.load()
    # split the document into chunks 
    st.session_state.text_splitter = CharacterTextSplitter(chunk_size = 100, chunk_overlap =0)
    st.session_state.chunked_doc = st.session_state.text_splitter.split_documents(st.session_state.docs)
    # Store chunked document into database
    st.session_state.vectors = FAISS.from_documents(st.session_state.chunked_doc, st.session_state.embeddings)


# set title of page
st.title('Document Information Search')
# load the model
llm_model  = HuggingFaceHub(repo_id = "mistralai/Mistral-7B-Instruct-v0.2", model_kwargs = {'top_k':1, 'temperature': 1, 'max_length': 64})

# set prompt template
prompt_template = """
[INST] Instruction: Answer the question based on your expertise in movies. Here is context to help:

{context}

QUESTION:
{input} [/INST]
"""

# set prompt template using defined prompt
prompt = PromptTemplate(input_variables = ['context', 'input'], template = prompt_template)

# generate chain
llm_chain = LLMChain(llm = llm_model, prompt = prompt)

# retrieve vectors 
retriever2 = st.session_state.vectors.as_retriever()
# create chain for passing list of documents to model
document_chain = create_stuff_documents_chain(llm_model, prompt)
# create retrival chain that retrieves documents
retrieval_chain = create_retrieval_chain(retriever2, document_chain)

# input text 
user_prompt = st.text_input('Input topic')
if user_prompt:
    # get response using retrival chain
    response = retrieval_chain.invoke({'input': user_prompt})
    # display response
    st.write(response['answer'])


