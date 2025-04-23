#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.agents import create_csv_agent
from tempfile import NamedTemporaryFile
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent

# Load environment variables from .env file
load_dotenv()

# Set page config for Streamlit
st.set_page_config(page_title="Wise - Ask your Questions")

# Header
st.header("Ask Your Questions")

# File uploaders
pdf = st.sidebar.file_uploader("Upload Your PDF", type='pdf')
file = st.sidebar.file_uploader("Upload Your CSV", type='csv')

# Handle CSV file
if file is not None:
    df = pd.read_csv(file)
    user_questions = st.text_input("Ask your CSV data")
    
    # Create agent for CSV data
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    if user_questions:
        response = agent.run(user_questions)
        st.write(response)

# Handle PDF file
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create knowledge base from text chunks
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Ask questions
    user_questions = st.text_input("Ask questions to your PDF data")
    if user_questions:
        docs = knowledge_base.similarity_search(user_questions)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_questions)
        st.write(response)

if __name__ == "__main__":
    main()


# In[ ]:




