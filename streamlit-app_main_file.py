import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Function to initialize the QA chain
def initialize_qa_chain(api_key):
    loader = DirectoryLoader(path='data/', glob="./*final.csv", loader_cls=CSVLoader)
    documents = loader.load()

    # Splitting the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Embed and store the texts
    persist_directory = 'db'
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    vectordb.persist()

    # Load the persisted database from disk
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    turbo_llm = ChatOpenAI(temperature=0,
                           model_name='gpt-3.5-turbo-16k',
                           openai_api_key=api_key)

    # Create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)
    return qa_chain

# Streamlit UI
def main():
    st.title('Langchain Retrieval QA Example')

    # User inputs OpenAI key
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    if openai_api_key:
        # Initialize environment
        with st.spinner('Initializing environment...'):
            qa_chain = initialize_qa_chain(openai_api_key)
            st.session_state['qa_chain'] = qa_chain
            st.success('Environment initialized!')
    
        # Query input
        query = st.text_input("Enter your query here")

        if st.button('Search') and query:
            with st.spinner('Searching for answers...'):
                # Prepare the prompt
                prompt = f'''
                You are an intelligent AI bot with the capability to search based on user query and look in the database\
                to identify all relevant offers for the same. If you don't find offers, then return all the information you find related to the query.

                Here is the user query in curly braces:
                {query}
                '''
                llm_response = st.session_state['qa_chain'](prompt)
                st.write(llm_response['result'])

if __name__ == "__main__":
    if 'qa_chain' not in st.session_state:
        st.session_state['qa_chain'] = None
    main()
