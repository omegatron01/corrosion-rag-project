# ─────────────────────────────────────────────────────────────
# app.py — Hugging Face Spaces deployment
# Uses Flan-T5 instead of Gemma (runs on free CPU)
# Vision model + FAISS + Flan-T5 = full RAG pipeline
# ─────────────────────────────────────────────────────────────

import os
import sys
import tempfile
import torch
import torch.nn.functional as F
import streamlit as st
import faiss
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Metal Corrosion Assessment System",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .result-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00c2a8;
    }
    .grade-safe {
        background-color: #1a472a;
        color: #69db7c;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .grade-warning {
        background-color: #4a3000;
        color: #ffd43b;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .grade-danger {
        background-color: #4a0000;
        color: #ff6b6b;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #00c2a8;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3748;
    }
    .standard-pill {
        display: inline-block;
        background-color: #1a2040;
        color: #90cdf4;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        margin: 4px 4px 4px 0;
        border: 1px solid #2d4a7a;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "model", "best_corrosion_model.pth")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "vector_store", "faiss_index.bin")
CHUNKS_PATH      = os.path.join(BASE_DIR, "vector_store", "chunks.pkl")

CLASS_NAMES = ["CORROSION", "NO_CORROSION"]

SOURCE_NAMES = {
    "iso_8501_rust_grades.txt"   : "ISO 8501-1 Rust Grade Definitions",
    "astm_b117_acceptance.txt"   : "ASTM B117 Salt Spray Testing Standard",
    "nace_sp0169_pipelines.txt"  : "NACE SP0169 Pipeline Corrosion Control",
    "remediation_guidelines.txt" : "Corrosion Remediation Guidelines"
}

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ══════════════════════════════════════════════════════════════
# Model Loading — cached so they load only once
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_vision_model():
    device = torch.device("cpu")
    model  = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()
    return model, device


@st.cache_resource
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        data = pickle.load(f)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, data["chunks"], data["metadata"], embedding_model


@st.cache_resource
def load_flan_t5():
    # Flan-T5-base — small, fast, runs on CPU
    # Perfect for structured assessment responses
    print("Loading Flan-T5...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model     = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-base"
    )
    model.eval()
    print("Flan-T5 ready!")
    return tokenizer, model


# ══════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════

def predict_corrosion(image, model, device):
    tensor = val_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs       = model(tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_class  = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()

    if predicted_class.lower() == "corrosion":
        if confidence_score >= 0.90:
            grade    = "Grade C"
            severity = "danger"
        elif confidence_score >= 0.70:
            grade    = "Grade B"
            severity = "warning"
        else:
            grade    = "Grade A"
            severity = "warning"
    else:
        grade    = "No Corrosion Detected"
        severity = "safe"

    return {
        "corrosion_grade" : grade,
        "corrosion_type"  : "surface corrosion",
        "confidence"      : confidence_score,
        "raw_prediction"  : predicted_class,
        "severity"        : severity,
        "probabilities"   : {
            CLASS_NAMES[i]: round(probabilities[i].item(), 4)
            for i in range(len(CLASS_NAMES))
        }
    }


def retrieve_chunks(vision_result, index, all_chunks,
                    all_metadata, embedding_model, k=3):
    query = (
        f"Standards for {vision_result['corrosion_grade']} "
        f"{vision_result['corrosion_type']} on structural metal. "
        f"Fitness for service and recommended action."
    )
    query_vector = embedding_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k=k)

    chunks  = [all_chunks[i]             for i in indices[0]]
    sources = [all_metadata[i]["source"] for i in indices[0]]
    return chunks, sources


def ask_flan_t5(vision_result, chunks, tokenizer, flan_model):
    context = " ".join(chunks)

    prompt = (
        f"You are a corrosion engineering expert. "
        f"A metal component has been assessed with the following result: "
        f"Corrosion grade {vision_result['corrosion_grade']}, "
        f"type {vision_result['corrosion_type']}, "
        f"confidence {vision_result['confidence']*100:.0f}%. "
        f"Based on this context from international standards: {context[:800]} "
        f"Is this component fit for service? "
        f"State the verdict clearly, cite the standard, "
        f"and give a specific recommended action."
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        outputs = flan_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.3
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def format_sources(sources):
    seen = []
    for s in sources:
        name = SOURCE_NAMES.get(s, s)
        if name not in seen:
            seen.append(name)
    return seen


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

def main():

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.title("Corrosion Assessment System")
        st.markdown("---")
        st.markdown("""
**How it works:**

1. Upload a metal surface image
2. Vision model detects corrosion grade
3. RAG retrieves relevant standards
4. AI generates a cited assessment

**Standards Referenced:**
- ISO 8501-1
- ASTM B117
- NACE SP0169
        """)
        st.markdown("---")
        st.caption("MobileNetV2 + FAISS + Flan-T5")

    # ── Header ─────────────────────────────────────────────────
    st.title("Metal Corrosion Assessment System")
    st.markdown(
        "Upload a photo of any metal surface to receive an automated "
        "corrosion grade and fitness-for-service recommendation based "
        "on international engineering standards."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a metal surface image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image",
                     use_column_width=True)
            assess_btn = st.button(
                "Assess Corrosion",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("Upload an image to begin assessment")
            assess_btn = False

    with col2:
        st.markdown("### Assessment Results")

        if uploaded_file and assess_btn:

            # Load all models
            with st.spinner("Loading models..."):
                vision_model, device       = load_vision_model()
                index, chunks, metadata, embed = load_faiss()
                flan_tokenizer, flan_model = load_flan_t5()

            # Vision model
            with st.spinner("Analysing image..."):
                vision_result = predict_corrosion(
                    image, vision_model, device
                )

            # Display vision result
            st.markdown(
                '<p class="section-header">Vision Model Result</p>',
                unsafe_allow_html=True
            )

            grade    = vision_result["corrosion_grade"]
            severity = vision_result["severity"]
            st.markdown(
                f'<span class="grade-{severity}">{grade}</span>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            confidence = vision_result["confidence"]
            st.markdown("**Confidence**")
            st.progress(confidence)
            st.caption(f"{confidence*100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            for cls, prob in vision_result["probabilities"].items():
                label = cls.replace("_", " ").title()
                st.markdown(f"**{label}:** {prob*100:.1f}%")

            st.markdown("---")

            # RAG retrieval
            with st.spinner("Retrieving standards..."):
                ret_chunks, ret_sources = retrieve_chunks(
                    vision_result, index, chunks,
                    metadata, embed
                )

            # Flan-T5 assessment
            with st.spinner("Generating assessment..."):
                answer = ask_flan_t5(
                    vision_result, ret_chunks,
                    flan_tokenizer, flan_model
                )

            # Display assessment
            st.markdown(
                '<p class="section-header">Engineering Assessment</p>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="result-card">{answer}</div>',
                unsafe_allow_html=True
            )

            # Standards
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p class="section-header">Standards Referenced</p>',
                unsafe_allow_html=True
            )
            for name in format_sources(ret_sources):
                st.markdown(
                    f'<span class="standard-pill">📋 {name}</span>',
                    unsafe_allow_html=True
                )

        elif not uploaded_file:
            st.markdown(
                '<div class="result-card">'
                'Upload a metal surface image and click '
                'Assess Corrosion to receive your engineering assessment.'
                '</div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()