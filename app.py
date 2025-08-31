import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import PyPDF2

# Load environment variables from .env file
load_dotenv()
# Ensure CHROMA_OPENAI_API_KEY is set if not already
openai_key = os.getenv("OPENAI_API_KEY")
if not os.getenv("CHROMA_OPENAI_API_KEY") and openai_key:
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)
# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

# FastAPI app
app = FastAPI(title="RAG Chat API")

class QuestionRequest(BaseModel):
    question: str

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("Loading documents from directory...")
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
        elif filename.endswith(".pdf"):
            pdf_text = extract_pdf_text(file_path)
            if pdf_text.strip():
                documents.append({"id": filename, "text": pdf_text})
            else:
                print(f"Warning: No text extracted from {filename}")
                
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Function to generate embeddings using OpenAI API
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding


# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks

# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        relevant_chunks = query_documents(request.question)
        answer = generate_response(request.question, relevant_chunks)
        return {"answer": answer.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize documents on startup
def initialize_documents():
    try:
        # Check if collection already has documents
        existing_count = collection.count()
        if existing_count > 0:
            print(f"Documents already loaded: {existing_count} chunks in database")
            return
            
        # Load documents from the directory
        directory_path = "./news_articles"
        documents = load_documents_from_directory(directory_path)
        
        if not documents:
            print("No documents found in news_articles directory")
            return

        print(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        chunked_documents = []
        for doc in documents:
            chunks = split_text(doc["text"])
            print(f"Splitting {doc['id']} into chunks")
            for i, chunk in enumerate(chunks):
                chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

        print(f"Split documents into {len(chunked_documents)} chunks")

        # Generate embeddings and store in ChromaDB
        for i, doc in enumerate(chunked_documents):
            print(f"Processing chunk {i+1}/{len(chunked_documents)}")
            doc["embedding"] = get_openai_embedding(doc["text"])
            collection.upsert(
                ids=[doc["id"]], 
                documents=[doc["text"]], 
                embeddings=[doc["embedding"]]
            )

        print("Document initialization completed!")
        
    except Exception as e:
        print(f"Error initializing documents: {e}")

# Run initialization when the app starts
initialize_documents()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
