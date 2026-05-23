import os
import faiss
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
CHUNKS_PATH      = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SOURCE_NAMES = {
    "iso_8501_rust_grades.txt"   : "ISO 8501-1 Rust Grade Definitions",
    "astm_b117_acceptance.txt"   : "ASTM B117 Salt Spray Testing Standard",
    "nace_sp0169_pipelines.txt"  : "NACE SP0169 Pipeline Corrosion Control",
    "remediation_guidelines.txt" : "Corrosion Remediation Guidelines"}

class CorrosionRAG:
    """
    Encapsulates the full RAG query pipeline.
    Load once, query many times.

    Usage:
        rag = CorrosionRAG(gemma_model, tokenizer)
        result = rag.query(vision_result)
    """

    def __init__(self, gemma_model, tokenizer):
        self.gemma_model     = gemma_model
        self.tokenizer       = tokenizer
        self.embedding_model = None
        self.index           = None
        self.all_chunks      = None
        self.all_metadata    = None
        self._load_resources()

    def _load_resources(self):
        """Load FAISS index, chunks, and embedding model into memory."""

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run ingest.py first.")

        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        print(f" {self.index.ntotal} chunks loaded")

        print("Loading chunks and metadata...")
        with open(CHUNKS_PATH, "rb") as f:
            data = pickle.load(f)
        self.all_chunks   = data["chunks"]
        self.all_metadata = data["metadata"]
        print(f"{len(self.all_chunks)} chunks ready")

        print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model ready")

    def _build_query(self, vision_result):
        """Turn vision model output into a search question."""
        return (
            f"International standards for {vision_result['corrosion_grade']} "
            f"{vision_result['corrosion_type']} on structural metal. "
            f"Fitness for service assessment and recommended action.")

    def _retrieve(self, query, k=3):
        """
        Embed the query and search FAISS for top-k matching chunks.
        Returns (chunks, sources).
        """
        query_vector = self.embedding_model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k=k)

        chunks  = [self.all_chunks[i]              for i in indices[0]]
        sources = [self.all_metadata[i]["source"]  for i in indices[0]]

        return chunks, sources

    def _build_prompt(self, vision_result, context):
        """Build the full prompt for Gemma."""
        return f"""<start_of_turn>user
You are a corrosion engineering expert assistant.
Do not use any Markdown formatting, asterisks, or bold text. Write in plain text only.
Assess whether this metal component is fit for service based on the
detected corrosion and the international standards provided below.

Always:
- State clearly if the component is fit for service or not
- Cite which standard supports your answer
- Give a specific recommended action

VISION MODEL DETECTION:
- Corrosion Grade: {vision_result['corrosion_grade']}
- Corrosion Type:  {vision_result['corrosion_type']}
- Confidence:      {vision_result['confidence'] * 100:.0f}%

RETRIEVED STANDARDS:
{context}

What is your assessment and recommendation?
<end_of_turn>
<start_of_turn>model
"""

    def _run_gemma(self, prompt):
        """Send prompt to Gemma and return decoded response."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.gemma_model.device)

        with torch.no_grad():
            output = self.gemma_model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer      = full_output.split("model")[-1].strip()
        return answer

    def _format_sources(self, sources):
        """Remove duplicates and map to professional names."""
        seen = []
        for s in sources:
            name = SOURCE_NAMES.get(s, s)
            if name not in seen:
                seen.append(name)
        return seen

    def query(self, vision_result):
        """
        Full RAG pipeline — call this with vision model output.

        Input:
            vision_result = {
                "corrosion_grade": "Grade C",
                "corrosion_type" : "surface corrosion",
                "confidence"     : 0.91
            }

        Output:
            {
                "answer" : "The component is not fit for service...",
                "sources": ["ISO 8501-1 Rust Grade Definitions", ...]
            }
        """

        # Step 1 — build search query
        query = self._build_query(vision_result)

        # Step 2 — retrieve relevant chunks
        chunks, sources = self._retrieve(query, k=3)

        # Step 3 — format context (no source labels)
        context = "\n".join(chunks)

        # Step 4 — build prompt
        prompt = self._build_prompt(vision_result, context)

        # Step 5 — run Gemma
        answer = self._run_gemma(prompt)

        # Step 6 — format sources professionally
        formatted_sources = self._format_sources(sources)

        return {
            "answer" : answer,
            "sources": formatted_sources
        }