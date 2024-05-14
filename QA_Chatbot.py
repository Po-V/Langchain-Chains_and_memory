from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline 
from langchain import PromptTemplate
import streamlit as st



st.set_page_config(page_title="Chatbot with RAG", layout="centered")
st.markdown("""
    <div style='background-color: #F5F5F5; padding: 16px; border-radius: 8px;'>
        <h1 style='color: #4285F4; text-align: center;'>Chatbot with RAG</h1>
    </div>
""", unsafe_allow_html=True)
st.write("##")


@st.cache_resource
def load_model_and_tokenizers(model_name):
    """
    Load pre-trained LLM model and tokenizer from HuggingFace
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512, padding='max_length', truncation=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def create_load_vectorstore(docs_text, _embeddings):
    """
    Create FAISS vector store and save locally. If index cannot be loaded,
    throw an exception.
    """
    vector_store = FAISS.from_texts(docs_text, _embeddings)
    vector_store.save_local(folder_path="../database/faiss_db", index_name="rag-DB")

    try:
        vector_store = FAISS.load_local(folder_path="../database/faiss_db",embeddings= _embeddings, index_name="rag-DB", 
                                    allow_dangerous_deserialization= True)
        print('Faiss index loaded ')
    except Exception as e:
        print("Fiass index loading failed \n",e)
    
    return vector_store


@st.cache_data
def generate_document_chunks(resource):
    """
    Instantiate embedding model and convert documents to chunks
    """
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    loader=WebBaseLoader(resource)
    docs= loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap =0)
    chunked_doc = text_splitter.split_documents(docs_transformed)
    docs_text = [doc.page_content for doc in chunked_doc]
    
    return docs_text, embeddings


@st.cache_data
def create_prompt_template():
    """
    Generate prompt template
    """

    template = """SYSTEM: You are an intelligent assistant helping the users with their questions on Langchain.

    Question: {question}

    Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

    Do not try to make up an answer:
    - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
    - If the context is empty, just say "I do not know the answer to that."

    =============
    {context}
    =============

    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate(template = template, input_variables = ['context', 'question'])
    return prompt


@st.cache_resource
def create_qa_chain(_model, _tokenizer, _vector_store, _prompt):
    """
    Create a retrieval QA chain with the given model, tokenizer and vector store
    """
    pipe = pipeline("text-generation", model= _model, tokenizer= _tokenizer, max_length=1024, padding='max_length', truncation = True)
    hf_pipeline = HuggingFacePipeline(pipeline = pipe)
    qa_chain = RetrievalQA.from_chain_type(llm = hf_pipeline, chain_type = 'stuff', 
                                           retriever = _vector_store.as_retriever(search_kwargs={'k': 2}), 
                                           chain_type_kwargs = {'prompt': _prompt},verbose= False)
    return qa_chain


def main():
    """
    Main function to run the Streamlit app
    """

    model_name = 'openai-community/gpt2'
    resource = 'https://www.langchain.com/langsmith'

    tokenizer, model = load_model_and_tokenizers(model_name)
    docs_text, embeddings = generate_document_chunks(resource)
    vector_store = create_load_vectorstore(docs_text, embeddings)
    prompt = create_prompt_template()
    qa_chain = create_qa_chain(model, tokenizer, vector_store, prompt)

    user_input = st.text_input("Ask your question here:")

    if user_input:
        result = qa_chain({'query': user_input})
        st.markdown("## Answer:")
        st.write(result['result'])


if __name__ == "__main__":
    main()


