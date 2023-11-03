# Importing necessary libraries
import streamlit as st
import pandas as pd

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

openai_api_key = "your-open-ai-key"
embedding = OpenAIEmbeddings(openai_api_key = openai_api_key)


persist_directory = 'db'
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo-16k',
    openai_api_key = openai_api_key
)

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)


def process_llm_response(llm_response):
    print(llm_response['result'])
    

# full example
query = "Amazon"
prompt = f'''
You are an intelligent AI bot with capability to search based on user query and look in the database\
to identify all relevant offers for the same. If you dont find offers, then return all the information you find related to the query.\

Consider below instructions for instance\

Instruction:
•	If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category.
•	If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand.
•	If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer.


Here is the user query in curly braces:
{query}

'''

llm_response = qa_chain(prompt)
process_llm_response(llm_response)
