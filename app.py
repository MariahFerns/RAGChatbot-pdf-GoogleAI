# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Install required librares
# # !pip install -r requirements.txt

# +
# Import required libraries
from PyPDF2 import PdfReader
import streamlit as st
import google.generativeai as genai
# import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings('ignore')

# +
# Setup the Streamlit page

st.set_page_config(page_title='🔖Document Chatbot', layout='wide')

st.markdown('''
## 🔖Document Chatbot: Get instant insights from your PDF's

This chatbot is built using Retrieval-Augmented Generation (RAG) framework, leveraging Google's
Generative AI model Gemini-Pro. 

### How it works
Follow the instructions below:

1) **Enter your API key** : You'll need a Google API key so that the chatbot can use Google's Generative AI models. Get your API key here: https://makersuite.google.com/app/apikey

2) **Upload your documents** : You'll can upload multiple PDF's at a time and the chatbot can answer questions based on any of the uploaded PDF's.

3) Ask a question : After processing the documents, ask any question related to the content of the uploaded documents.
''')

api_key = st.text_input('Enter your Google API Key:', type='password', key='api_key_input')



# +
# Define function to read input PDF and convert to text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

# Define function for splitting text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return text_splitter.create_documents(chunks)

# Define a function to convert text chunks into embeddings and store in vector database
def get_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory = "./chroma_db")
    vector_store.persist()
    
# Pass information to LLM
def get_conversational_chain():
    prompt_template = ''' Answer the question as detailed as possible from the provided context, make sure to provide all the details.
                    If the answer is not available in the documents, don't provide wrong answer.\n\n
                    Context: \n {context} \n
                    Question: \n {question} \n
                    
                    Answer:
                    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, prompt=prompt, chain_type='stuff')
    return chain
    

# Define function to process user qestion, search most relevant docs based on user question and generate response using conversational chain
def user_input(user_question, api_key):
    # Instantiate embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
    # Fetch vector DB from local
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # Get embeddings similar to user question
    context = vector_db.similarity_search(user_question)
    # Instantiate conversational chain
    chain = get_conversational_chain()
    # Get response
    response = chain({'input_documents': context, 'question': user_question}, return_only_outputs=True)
    # Print response
    st.write('Reply: ', response['output_text'])
    


# +
# Create UI for streamlit

def main():
    st.header('Document Chatbot')
    
    user_question = st.text_input('Ask a question from the PDF files', key='user_question')
    
    # Ensure API key and user question are provided
    if user_question and api_key:
        user_input(user_question, api_key)
    
    with st.sidebar:
        st.title('Menu')
        # Upload PDF input docs
        pdf_docs = st.file_uploader('Upload your PDF files', accept_multiple_files=True, key='pdf_uploader')
        
         # Check if API key is provided before processing
        if st.button('Submit', key='process_button') and api_key:
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success('Done')
        


# -

if __name__ == '__main__':
    main()
