import os
import faiss
import numpy as np
import pickle
import torch
import urllib.request
import urllib.parse
import json
from sentence_transformers import SentenceTransformer

base_dir         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_store_dir = os.path.join(base_dir, "vector_store")
faiss_index_path = os.path.join(vector_store_dir, "faiss_index.bin")
chunks_path      = os.path.join(vector_store_dir, "chunks.pkl")

embedding_model_name = "all-MiniLM-L6-v2"

source_names = {
    "iso_8501_rust_grades.txt"   : "ISO 8501-1 Rust Grade Definitions",
    "astm_b117_acceptance.txt"   : "ASTM B117 Salt Spray Testing Standard",
    "nace_sp0169_pipelines.txt"  : "NACE SP0169 Pipeline Corrosion Control",
    "remediation_guidelines.txt" : "Corrosion Remediation Guidelines"}

class faiss_search_tool:
    """
    Searches the local FAISS vector database.
    Contains ISO, ASTM, and NACE standards documents.
    No internet connection required.
    """

    def __init__(self):
        # Load FAISS index
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_index_path}. "
                "Run ingest.py first."
            )

        print("  Loading FAISS index...")
        self.index = faiss.read_index(faiss_index_path)

        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        self.chunks   = data["chunks"]
        self.metadata = data["metadata"]

        self.embed = SentenceTransformer(embedding_model_name)
        print(f"  FAISS tool ready: {self.index.ntotal} chunks")

    def run(self, query, k=3):
        """
        Embeds query and searches for top-k matching chunks.
        Returns dict with results and sources.
        """
        query_vector = self.embed.encode([query]).astype("float32")
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k=k)

        results = []
        raw_sources = []

        for i in indices[0]:
            results.append(self.chunks[i])
            raw_sources.append(self.metadata[i]["source"])

        seen = []
        for s in raw_sources:
            name = source_names.get(s, s)
            if name not in seen:
                seen.append(name)

        return {
            "success" : True,
            "tool"    : "Local Standards Knowledge Base",
            "results" : results,
            "sources" : seen}

def wikipedia_search_tool(query):
    """
    Searches Wikipedia using their free REST API.
    Returns a plain text summary of the most relevant article.
    No API key needed.
    """
    try:
        encoded_query = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/api/rest_v1/"
            f"page/summary/{encoded_query}")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CorrosionRAG/1.0"})

        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())

        if "extract" in data and data["extract"]:
            return {
                "success" : True,
                "tool"    : "Wikipedia",
                "title"   : data.get("title", query),
                "content" : data["extract"][:800]}

        return {
            "success" : False,
            "tool"    : "Wikipedia",
            "content" : ""}

    except Exception as e:
        return {
            "success" : False,
            "tool"    : "Wikipedia",
            "content" : ""}

class tool_router:
    """
    Orchestrates all available tools.
    Runs each tool, collects results, combines into
    a single context string for Gemma to reason over.
    """

    def __init__(self, faiss_tool):
        self.faiss_tool = faiss_tool

    def run(self, vision_result):
        """
        Runs all tools relevant to the vision result.
        Returns (combined_context, tools_used).

        combined_context — single string fed to Gemma
        tools_used       — list of tool names for display
        """
        grade         = vision_result["corrosion_grade"]
        corrosion_type = vision_result["corrosion_type"]

        combined_context = ""
        tools_used       = []

        print("  Tool 1: Searching local standards database...")
        faiss_query = (
            f"Standards for {grade} {corrosion_type} "
            f"on structural metal. "
            f"Fitness for service assessment and recommended action."
        )
        faiss_result = self.faiss_tool.run(faiss_query)

        if faiss_result["success"]:
            tools_used.extend(faiss_result["sources"])
            combined_context += "\n[LOCAL STANDARDS DATABASE]\n"
            for chunk in faiss_result["results"]:
                combined_context += f"{chunk}\n"

        print("  Tool 2: Searching Wikipedia...")
        wiki_query  = f"ISO 8501 {grade} rust corrosion structural steel"
        wiki_result = wikipedia_search_tool(wiki_query)

        if wiki_result["success"]:
            tools_used.append(f"Wikipedia: {wiki_result['title']}")
            combined_context += f"\n[WIKIPEDIA: {wiki_result['title']}]\n"
            combined_context += wiki_result["content"] + "\n"
        else:
            print("  Tool 2: Wikipedia unavailable, continuing without it")

        return combined_context, tools_used


class CorrosionRAG:
    """
    Full agentic RAG pipeline.
    Combines MCP-style tool routing with Gemma reasoning.

    Usage:
        rag = CorrosionRAG(gemma_model, tokenizer)
        result = rag.query(vision_result)
    """

    def __init__(self, gemma_model, tokenizer):
        self.gemma_model = gemma_model
        self.tokenizer   = tokenizer

        print("Initialising RAG pipeline...")

        self.faiss_tool = faiss_search_tool()

        self.router = tool_router(self.faiss_tool)

        print("RAG pipeline ready!")

    def _build_prompt(self, vision_result, combined_context):
        """Builds the full prompt for Gemma."""
        return f"""<start_of_turn>user
You are a corrosion engineering expert assistant.
Do not use any Markdown formatting, asterisks, or bold text.
Write in plain text only.

You have been given information from multiple sources.
Use all of it to assess whether this metal component is fit for service.

Always:
- State clearly if the component is fit for service or not
- Cite which source or standard supports your answer
- Give a specific recommended action
- Be concise and professional

VISION MODEL DETECTION:
- Corrosion Grade: {vision_result['corrosion_grade']}
- Corrosion Type:  {vision_result['corrosion_type']}
- Confidence:      {vision_result['confidence'] * 100:.0f}%

INFORMATION FROM TOOLS:
{combined_context}

What is your assessment and recommendation?
<end_of_turn>
<start_of_turn>model
"""

    def _run_gemma(self, prompt):
        """Sends prompt to Gemma and returns decoded response."""
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
                pad_token_id=self.tokenizer.eos_token_id)

        full_output = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True)
        answer = full_output.split("model")[-1].strip()
        return answer

    def query(self, vision_result):
        """
        Full agentic RAG pipeline.

        Input:
            vision_result = {
                "corrosion_grade" : "Grade C",
                "corrosion_type"  : "surface corrosion",
                "confidence"      : 0.91
            }

        Output:
            {
                "answer"  : "The component is not fit for service...",
                "sources" : ["ISO 8501-1...", "Wikipedia: Rust..."]
            }
        """
        print("Running tool router...")

        combined_context, tools_used = self.router.run(vision_result)
        prompt = self._build_prompt(vision_result, combined_context)
        print("  Running Gemma...")
        answer = self._run_gemma(prompt)
        seen = []
        for t in tools_used:
            if t not in seen:
                seen.append(t)

        return {
            "answer"  : answer,
            "sources" : seen}