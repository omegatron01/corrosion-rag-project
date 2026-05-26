import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import tempfile
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query import CorrosionRAG

st.set_page_config(
    page_title="Metal Corrosion Assessment System",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }

    /* Result cards */
    .result-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00c2a8;
    }

    /* Grade badges */
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

    /* Confidence bar label */
    .confidence-label {
        font-size: 14px;
        color: #a0aec0;
        margin-bottom: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #00c2a8;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3748;
    }

    /* Standard citation pill */
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

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "best_corrosion_model.pth")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["CORROSION", "NO_CORROSION"]

@st.cache_resource
def load_vision_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()
    return model.to(device), device


@st.cache_resource
def load_rag_pipeline():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    gemma     = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        quantization_config=bnb_config,
        device_map="auto"
    )
    return CorrosionRAG(gemma, tokenizer)


def predict_corrosion(image, model, device):
    tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs       = model(tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_class  = class_names[predicted_idx.item()]
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
            class_names[i]: round(probabilities[i].item(), 4)
            for i in range(len(class_names))
        }
    }



def main():

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/rust.png", width=80)
        st.title("Corrosion Assessment System")
        st.markdown("---")
        st.markdown("""
        **How it works:**

        1. Upload a metal surface image
        2. AI vision model detects corrosion grade
        3. RAG pipeline retrieves relevant international standards
        4. Gemma LLM generates a cited assessment

        **Standards Referenced:**
        - ISO 8501-1
        - ASTM B117
        - NACE SP0169
        """)
        st.markdown("---")
        st.caption("Powered by MobileNetV2 + Gemma 3 + FAISS")

    st.title("🔩 Metal Corrosion Assessment System")
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
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            assess_btn = st.button(
                "🔍 Assess Corrosion",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("👆 Upload an image to begin assessment")
            assess_btn = False

    with col2:
        st.markdown("### Assessment Results")

        if uploaded_file and assess_btn:

            # ── Load models ────────────────────────────────────
            with st.spinner("Loading vision model..."):
                vision_model, device = load_vision_model()

            # ── Vision model prediction ────────────────────────
            with st.spinner("Analysing image..."):
                vision_result = predict_corrosion(image, vision_model, device)

            # ── Display vision results ─────────────────────────
            st.markdown(
                '<p class="section-header">Vision Model Result</p>',
                unsafe_allow_html=True
            )

            # Grade badge with color coding
            grade    = vision_result["corrosion_grade"]
            severity = vision_result["severity"]
            st.markdown(
                f'<span class="grade-{severity}">{grade}</span>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence bar
            confidence = vision_result["confidence"]
            st.markdown(
                '<p class="confidence-label">Confidence</p>',
                unsafe_allow_html=True
            )
            st.progress(confidence)
            st.caption(f"{confidence*100:.1f}%")

            # Probabilities
            st.markdown("<br>", unsafe_allow_html=True)
            probs = vision_result["probabilities"]
            for class_name, prob in probs.items():
                label = class_name.replace("_", " ").title()
                st.markdown(f"**{label}:** {prob*100:.1f}%")

            st.markdown("---")

            # ── RAG + Gemma ────────────────────────────────────
            with st.spinner("Retrieving standards and generating assessment..."):
                rag        = load_rag_pipeline()
                rag_result = rag.query(vision_result)

            # ── Display Gemma assessment ───────────────────────
            st.markdown(
                '<p class="section-header">Engineering Assessment</p>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="result-card">{rag_result["answer"]}</div>',
                unsafe_allow_html=True
            )

            # ── Standards referenced ───────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p class="section-header">Standards Referenced</p>',
                unsafe_allow_html=True
            )
            for source in rag_result["sources"]:
                st.markdown(
                    f'<span class="standard-pill">📋 {source}</span>',
                    unsafe_allow_html=True
                )

        elif not uploaded_file:
            st.markdown(
                '<div class="result-card">'
                'Upload a metal surface image and click Assess Corrosion '
                'to receive your engineering assessment.'
                '</div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()