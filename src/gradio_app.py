import os
import sys
import tempfile
import torch
import gradio as gr
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query import CorrosionRAG

base_dir        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path      = os.path.join(base_dir, "models", "best_corrosion_model.pth")
gemma_model_id  = "google/gemma-3-4b-it"

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["CORROSION", "NO_CORROSION"]


def load_vision_model():
    """
    Loads MobileNetV2 with your trained weights.
    Returns the model ready for inference.
    """
    print("\nLoading vision model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corrosion_model = models.mobilenet_v2(weights=None)
    num_features    = corrosion_model.classifier[1].in_features
    corrosion_model.classifier[1] = nn.Linear(num_features, 2)

    corrosion_model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    corrosion_model.eval()
    corrosion_model = corrosion_model.to(device)

    print(f"Vision model loaded on: {device}")
    return corrosion_model, device


def load_gemma():
    """
    Loads Gemma in 4-bit quantized mode to save GPU memory.
    Returns (tokenizer, gemma_model).
    """
    print("\nLoading Gemma...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(gemma_model_id)

    gemma_model = AutoModelForCausalLM.from_pretrained(
        gemma_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Gemma loaded!")
    return tokenizer, gemma_model

def predict_corrosion(image_path, corrosion_model, device):
    """
    Takes an image path, runs it through the vision model.
    Returns a structured dict for the RAG pipeline.
    """
    img          = Image.open(image_path).convert("RGB")
    input_tensor = val_test_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_outputs   = corrosion_model(input_tensor)
        probabilities = F.softmax(raw_outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_class  = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    if predicted_class.lower() == "corrosion":
        if confidence_score >= 0.90:
            grade = "Grade C"
        elif confidence_score >= 0.70:
            grade = "Grade B"
        else:
            grade = "Grade A"
    else:
        grade = "No corrosion detected"

    return {
        "corrosion_grade" : grade,
        "corrosion_type"  : "surface corrosion",
        "confidence"      : confidence_score,
        "raw_prediction"  : predicted_class,
        "all_probabilities": {
            class_names[i]: round(probabilities[i].item(), 4)
            for i in range(len(class_names))
        }
    }


def assess_metal_image(image, corrosion_model, device, rag):
    """
    Called by Gradio when user clicks Assess Corrosion.
    Returns two strings: vision output and Gemma assessment.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    vision_result = predict_corrosion(tmp_path, corrosion_model, device)
    os.unlink(tmp_path)

    probs = vision_result["all_probabilities"]
    prob_lines = "\n".join([
        f"  {k.replace('_', ' ').title()}: {v*100:.1f}%"
        for k, v in probs.items()
    ])

    vision_output = f"""VISION MODEL RESULT
-------------------
Prediction: {vision_result['raw_prediction'].replace('_', ' ').title()}
Corrosion Grade: {vision_result['corrosion_grade']}
Confidence: {vision_result['confidence']*100:.1f}%

Probabilities:
{prob_lines}
"""

    rag_result = rag.query(vision_result)

    standards = "\n".join([
        f"  - {name}" for name in rag_result["sources"]
    ])

    gemma_output = f"""GEMMA'S ASSESSMENT
{rag_result['answer']}

STANDARDS REFERENCED:
{standards}
"""

    return vision_output, gemma_output

def launch_app():
    """
    Loads all models and launches the Gradio interface.
    """
    print("\n" + "=" * 55)
    print("METAL CORROSION ASSESSMENT SYSTEM — STARTING UP")
    print("=" * 55)

    corrosion_model, device = load_vision_model()
    tokenizer, gemma_model  = load_gemma()
    rag = CorrosionRAG(gemma_model, tokenizer)

    print("\nAll models loaded. Launching interface...\n")

    def handler(image):
        return assess_metal_image(image, corrosion_model, device, rag)

    with gr.Blocks(title="Metal Corrosion Assessment System") as app:

        gr.Markdown("# Metal Corrosion Assessment System")
        gr.Markdown(
            "Upload a photo of a metal surface to receive an automated "
            "corrosion grade assessment and fitness-for-service recommendation "
            "based on international standards."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Metal Image")
                submit_btn  = gr.Button("Assess Corrosion", variant="primary")

            with gr.Column():
                vision_output = gr.Textbox(label="Vision Model Result", lines=10)
                gemma_output  = gr.Textbox(label="Gemma Assessment",    lines=18)

        submit_btn.click(
            fn=handler,
            inputs=image_input,
            outputs=[vision_output, gemma_output]
        )

    app.launch(share=True)

if __name__ == "__main__":
    launch_app()