import gradio as gr
from PIL import Image
import tempfile
# ── Build the Gradio interface ─────────────────────────────────
with gr.Blocks(title="Metal Corrosion Assessment System") as app:

    gr.Markdown("# Metal Corrosion Assessment System")
    gr.Markdown("Upload a photo of a metal surface to assess corrosion grade and fitness for service.")

    with gr.Row():
        # Left column — image upload
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Metal Image")
            submit_btn = gr.Button("Assess Corrosion", variant="primary")

        # Right column — results
        with gr.Column():
            vision_output = gr.Textbox(label="Vision Model Result", lines=7)
            gemma_output = gr.Textbox(label="Gemma Assessment", lines=15)

    # Connect button to function
    submit_btn.click(
        fn=assess_metal_image,
        inputs=image_input,
        outputs=[vision_output, gemma_output]
    )

# Launch the app
app.launch(share=True)  # share=True gives you a public lin