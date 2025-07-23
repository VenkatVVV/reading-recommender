from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import pandas as pd 
import os

DATASET = os.environ['DATASET_FILEPATH']
CURRENT_DIR = os.path.abspath(__file__)

def load_environment(environment_path):
    load_dotenv(environment_path)

def get_data(filename): 

    data = pd.read_csv(f"{DATASET}/{filename}")
    return data

def tag_description(data):

    # joins the ISBNs of each book item with corresponding description
    data['tagged_description'] = data[['isbn13', 'description']].astype(str).agg(" ".join, axis=1)
    data['tagged_description'].to_csv(f"{DATASET}/tagged_description.txt")

def load_and_split(data,size= 0, overlap = 0):

    tag_description(data)
    # Load the data
    raw_descriptions = TextLoader(f"{DATASET}/tagged_description.txt").load()
    
    # Define text splitter
    text_splitter = CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,seperator="\n")
    
    # split documents
    descriptions = text_splitter.split_documents(raw_descriptions)

    return descriptions

def embed_descriptions(descriptions, embedding_model="models/embedding-001"):

    vector_database = Chroma.from_documents(
        documents=descriptions,
        embedding=GoogleGenerativeAIEmbeddings(model=embedding_model),
        persist_directory=f"{CURRENT_DIR}/chroma_db"
    )
    
    return vector_database

def retrieve_semantic_recommenadtions(data, vector_database, query, top_k=10): 
    '''
    Uses similarity search and returns top k recommendations in a pd Dataframe 
    '''

    recommendations = vector_database.similarity_search(query, k=50)
    recommendations_list = []
    for i in range(len(recommendations)):
        recommendations_list += [int(recommendations[i].page_content.strip('"').split()[0])]

    return data[data["isbn13"].isin(recommendations_list)].head(top_k) 







