
# imports
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain import HuggingFaceHub
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

# loading hf api key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_API_KEY')

# set the model to use
model_name = 'HuggingFaceH4/zephyr-7b-beta'

# loading model from HF Hub
llm_model = HuggingFaceHub(repo_id = model_name, model_kwargs = {'top_k':50, 'temperature': 0.8, 'max_length': 512})

# prompt template
template = """<s>[INST] 
You are a helpful AI assistant who answers questions in short sentences. Here is context to help: {chat_history}. 

Question:
{input}

Chatbot:"""


# set the prompt template 
prompt = PromptTemplate(
    input_variables=["chat_history", "input"], template=template
)

# add memory with memory key
memory = ConversationBufferMemory(memory_key="chat_history")

# create llm chain that takes user input, formats it using prompt_template, and pass formated prompt to LLM with memory
llm_chain = LLMChain(
    llm=llm_model,
    prompt=prompt,
    verbose=False,
    memory=memory,
    output_key = 'answer'
)

# make predictions for user input
print(llm_chain.predict(input="Hi there my friend"))
print(llm_chain.predict(input = "I am well, how are you?"))

