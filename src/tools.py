
import urllib.request
import urllib.parse
import json
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

def search_wikipedia(query, sentences=5):
    """
    Searches Wikipedia for engineering standards information.
    Returns a plain text summary.
    Free, no API key needed.
    """
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CorrosionRAG/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())

        if "extract" in data:
            return {
                "success": True,
                "source" : "Wikipedia",
                "title"  : data.get("title", query),
                "content": data["extract"][:1000]
            }
        return {"success": False, "content": ""}

    except Exception as e:
        return {"success": False, "content": f"Wikipedia search failed: {e}"}

class FAISSSearchTool:
    """
    Searches your local FAISS vector database.
    This is your existing knowledge base — ISO, ASTM, NACE docs.
    """

    def __init__(self, index_path, chunks_path):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        self.chunks   = data["chunks"]
        self.metadata = data["metadata"]
        self.embed    = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query, k=3):
        query_vector = self.embed.encode([query]).astype("float32")
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k=k)

        results = []
        for i in indices[0]:
            results.append({
                "content": self.chunks[i],
                "source" : self.metadata[i]["source"]
            })
        return {
            "success": True,
            "source" : "Local Standards Knowledge Base",
            "results": results
        }

class ToolRouter:
    """
    Decides which tools to call based on the vision result.
    Runs all relevant tools and combines their output.
    This is the MCP-style orchestration layer.
    """

    def __init__(self, faiss_tool):
        self.faiss_tool = faiss_tool
        self.tools_used = []

    def run(self, vision_result):
        """
        Runs all relevant tools for a given vision result.
        Returns combined context for Gemma.
        """
        self.tools_used = []
        combined_context = ""

        grade = vision_result["corrosion_grade"]
        ctype = vision_result["corrosion_type"]

        print("  Running tool: FAISS knowledge base search...")
        faiss_result = self.faiss_tool.search(
            f"Standards for {grade} {ctype} on structural metal "
            f"fitness for service assessment"
        )
        if faiss_result["success"]:
            self.tools_used.append("Local Standards Knowledge Base")
            combined_context += "\n[LOCAL STANDARDS DATABASE]\n"
            for r in faiss_result["results"]:
                combined_context += f"{r['content']}\n"

        print("  Running tool: Wikipedia search...")
        wiki_query = f"ISO 8501 {grade} rust corrosion steel"
        wiki_result = search_wikipedia(wiki_query)
        if wiki_result["success"]:
            self.tools_used.append("Wikipedia")
            combined_context += f"\n[WIKIPEDIA: {wiki_result['title']}]\n"
            combined_context += wiki_result["content"] + "\n"

        return combined_context, self.tools_used