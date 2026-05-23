import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
OUTPUT_DIR       = os.path.join(BASE_DIR, "vector_store")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.bin")
CHUNKS_PATH      = os.path.join(OUTPUT_DIR, "chunks.pkl")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 150   
OVERLAP    = 30   


def load_documents(knowledge_base_dir):
    """
    Reads all .txt files in the knowledge_base folder.
    Returns a list of (filename, text) tuples.
    """
    documents = []
    txt_files = [f for f in os.listdir(knowledge_base_dir) if f.endswith(".txt")]

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {knowledge_base_dir}")

    print(f"Found {len(txt_files)} knowledge base files:")
    for filename in txt_files:
        filepath = os.path.join(knowledge_base_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append((filename, text))
        print(f"{filename} ({len(text.split())} words)")

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Splits text into overlapping chunks of chunk_size words.
    Overlap ensures sentences at boundaries are never lost.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_knowledge_base():
    """
    Main function — runs the full ingestion pipeline:
    1. Load documents
    2. Chunk all documents
    3. Embed all chunks
    4. Build FAISS index
    5. Save index and chunks to disk
    """

    print("\n" + "=" * 55)
    print("CORROSION RAG — KNOWLEDGE BASE INGESTION")
    print("=" * 55)
    documents = load_documents(KNOWLEDGE_BASE_DIR)

    print(f"\nChunking documents (size={CHUNK_SIZE}, overlap={OVERLAP})...")

    all_chunks   = []  
    all_metadata = []  

    for filename, text in documents:
        chunks = chunk_text(text)
        print(f"  {filename} → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source"     : filename,
                "chunk_index": i
            })

    print(f"\nTotal chunks across all files: {len(all_chunks)}")
    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME})")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model ready!")

    print("\nEmbedding all chunks (this may take a minute)...")
    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True)

    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)
    print(f"Embeddings shape: {embeddings.shape}") 

    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]            
    index     = faiss.IndexFlatIP(dimension)   
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")

    print(f"\nSaving index to: {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"Saving chunks to: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({
            "chunks"  : all_chunks,
            "metadata": all_metadata
        }, f)

    print("\n" + "=" * 55)
    print(f"Ingestion complete!")
    print(f"   {len(all_chunks)} chunks stored")
    print(f"   Index saved to: vector_store/faiss_index.bin")
    print(f"   Chunks saved to: vector_store/chunks.pkl")

    return index, all_chunks, all_metadata

if __name__ == "__main__":
    build_knowledge_base()