import pandas as pd
import numpy as np 
from dotenv import load_dotenv 
import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

load_dotenv(dotenv_path="/Users/venkatvive/Documents/projects/reading-recommender/.env")

books = pd.read_csv("/Users/venkatvive/Documents/projects/reading-recommender/data/processed/books_categories_tones.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["thumbnail"].isna(), 
                                    "data/cover-not-found.jpg",
                                    books["large_thumbnail"],
                                    )

# Check if Chroma database already exists
chroma_dir = "/Users/venkatvive/Documents/projects/reading-recommender/vector-embedding/chroma_db"
if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
    # Load existing database
    db_books = Chroma(
        persist_directory=chroma_dir,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    print("Loaded existing Chroma database")
else:
    # Create new database
    raw_documents = TextLoader("/Users/venkatvive/Documents/projects/reading-recommender/data/processed/tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(
        documents=documents,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=chroma_dir
    )
    print("Created new Chroma database")

def get_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        top_k: int = 50,
        top_recommendations: int = 12,
) -> pd.DataFrame:
    
    recommendations = db_books.similarity_search(query, k=top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recommendations]
    book_recommendations = books[books['isbn13'].isin(books_list)].head(top_recommendations)

    if category != "All":
        book_recommendations = book_recommendations[book_recommendations['simple_categories'] == category].head(top_recommendations)
    else:
        book_recommendations = book_recommendations.head(top_recommendations)

    if tone == "Happy": 
        book_recommendations.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Surprising':
        book_recommendations.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == 'Angry':
        book_recommendations.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recommendations.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == "Sad":
        book_recommendations.sort_values(by='sadness', ascending=False, inplace=True)

    return book_recommendations

def recommend(
        query: str,
        category: str,
        tone: str,
):
    recommendations = get_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_description = " ".join(description.split()[:30])+"..."

        authors_split = row['authors'].split(";")
        if len(authors_split) == 2: 
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"],caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description",
                                placeholder = "E.g., A Suspenseful book about a crime")
        category_dropdown = gr.Dropdown(label = "Select a Genre", choices = categories, value = "All")
        tone_dropdown = gr.Dropdown(label = "Select an emotional Tone", choices = tones, value = "All")

        submit_button = gr.Button("Recommend!")

    gr.Markdown("## Recommended Books")
    output = gr.Gallery(label="Recommended Books", columns = 6, rows = 2, object_fit = "contain")

    submit_button.click(fn = recommend,
                             inputs = [user_query, category_dropdown, tone_dropdown],
                               outputs = output)
        
if __name__ == "__main__":
    dashboard.launch()