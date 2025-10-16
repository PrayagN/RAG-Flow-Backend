from fastapi.responses import StreamingResponse
import os
import re
import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import json


import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model and ChromaDB client
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

def sanitize_collection_name(name):
    base = os.path.splitext(name)[0]
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', base)
    sanitized = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', sanitized)
    if len(sanitized) < 3:
        sanitized = sanitized + "xyz"
    return sanitized

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text_content = ""
    if file.filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        except Exception as e:
            return {"file_id": None, "chunks": 0, "error": f"PDF parse failed: {str(e)}"}
    elif file.filename.lower().endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        except Exception as e:
            return {"file_id": None, "chunks": 0, "error": f"TXT parse failed: {str(e)}"}
    else:
        return {"file_id": None, "chunks": 0, "error": "Unsupported file type"}

    if not text_content.strip():
        return {"file_id": None, "chunks": 0, "error": "File parse failed (empty content)"}

    chunks = [text_content[i:i+500] for i in range(0, len(text_content), 500)]

    collection_name = sanitize_collection_name(file.filename)
    for collection in chroma_client.list_collections():
        if collection.name == collection_name:
            chroma_client.delete_collection(collection_name)
            break
    collection = chroma_client.create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        emb = embedding_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[f"{collection_name}_{i}"]
        )

    return {"file_id": collection_name, "chunks": len(chunks)}

@app.post("/ask")
async def ask(question: str = Form(...), file_id: str = Form(None)):
    if not file_id or file_id not in [c.name for c in chroma_client.list_collections()]:
        return {"answer": "File not found. Please upload your document first."}

    collection = chroma_client.get_collection(file_id)
    q_emb = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3,
        include=["documents"]
    )
    top_chunks = results.get("documents", [[]])[0]
    context = "\n---\n".join(top_chunks)

    # Build prompt for LLM
    prompt = (
        "You are an expert assistant. Use the below CONTEXT to answer the QUESTION.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"
    )

    # Call Ollama (Llama 2) running locally
    try:
        ollama_url = "http://localhost:11434/api/generate"
        payload = {"model": "llama2", "prompt": prompt, "stream": False}
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except Exception as e:
        answer = f"Error calling LLM: {str(e)}"

    return {"answer": answer}

@app.post("/ask-stream")
def ask_stream(question: str = Form(...), file_id: str = Form(None)):

    # Check file_id and collection existence
    if not file_id or file_id not in [c.name for c in chroma_client.list_collections()]:
        def error_stream():
            yield "File not found. Please upload your document first."
        return StreamingResponse(error_stream(), media_type="text/plain")

    collection = chroma_client.get_collection(file_id)
    q_emb = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3,
        include=["documents"]
    )
    top_chunks = results.get("documents", [[]])[0]
    context = "\n---\n".join(top_chunks)

    prompt = (
        "You are an expert assistant. Use the below CONTEXT to answer the QUESTION.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"
    )

    def stream_llama():
        ollama_url = "http://localhost:11434/api/generate"
        payload = {"model": "llama2", "prompt": prompt, "stream": True}
        with requests.post(ollama_url, json=payload, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode())
                        token = data.get("response", "")
                        yield token
                    except Exception as e:
                     print(f"Error in stream_llama: {str(e)}")
                     yield str(e)
 
    return StreamingResponse(stream_llama(), media_type="text/plain")