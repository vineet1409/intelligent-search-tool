# intelligent-search-tool using LLMs (openai embeddings, Langchain Agents and Chromdb vectordatabase)
Extracts the structured data and retrieves offers related to user's query based on products, retailers, etc.

## How to use?
1. Firstly run the "preprocessing" notebook under "notebooks" folder for processing and merging of datasets. The intent seems to be to create a unified dataset (df_merge_final) that provides a holistic view of each brand's product category, its associated offers, and the retailers providing those offers

2. Then run the "VectorDB_creation_ChromaDB_OpenAI" notebook under "notebooks" folder to create a vector-database for the dataset.

3. create a virtual environment and locally install all the requirements or dependencies by "**pip install -r requirements.tx**"

4. Run the python script to check if everything is set up properly "**python main_py_file.py**". The script uses vector databases for semantic search based on user's query.
   By default the query mentioned in script is "**amazon**". Be sure to add your openapi-key in the script.

5. Finally run the UI tool based on streamlit by "**streamlit run streamlit-app_main_file.py**".
