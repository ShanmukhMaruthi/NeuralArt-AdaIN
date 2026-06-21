
import torch
import gradio as gr

from src.encoder import VGGEncoder
from src.decoder import Decoder
from src.adain import adaptive_instance_normalization
from src.utils import preprocess_image, tensor_to_image


# ==========================================
# Device Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# Load Models
# ==========================================

VGG_PATH = "models/vgg_normalised.pth"
DECODER_PATH = "models/decoder_final.pth"


encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device
    )
)


encoder.eval()
decoder.eval()


# ==========================================
# Style Transfer Function
# ==========================================

def stylize(content_image, style_image, alpha):

    content_tensor = preprocess_image(
        content_image
    ).to(device)

    style_tensor = preprocess_image(
        style_image
    ).to(device)


    with torch.no_grad():

        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )


        target_features = adaptive_instance_normalization(
            content_features,
            style_features
        )


        target_features = (
            alpha * target_features
            +
            (1 - alpha) * content_features
        )


        generated_image = decoder(
            target_features
        )


    return tensor_to_image(
        generated_image
    )


# ==========================================
# Custom CSS
# ==========================================

css = """
.gradio-container {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
    font-family: Arial, sans-serif;
}

.hero-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #c084fc;
    text-shadow: 0 0 25px #9333ea;
}

.hero-subtitle {
    text-align: center;
    color: #e879f9;
    font-size: 20px;
}

.hero-description {
    text-align: center;
    color: #d8b4fe;
}

.divider {
    height: 3px;
    width: 200px;
    margin: 20px auto;
    background: linear-gradient(90deg, #7c3aed, #ec4899);
}

footer {
    display: none !important;
}
"""


# ==========================================
# Gradio Interface
# ==========================================

with gr.Blocks(
    css=css,
    title="NeuralArt AI - AdaIN Style Transfer"
) as demo:


    gr.HTML(
        """
        <div class="hero-title">
            🎨 NeuralArt AI
        </div>

        <div class="hero-subtitle">
            Transform Photos into Artistic Masterpieces
        </div>

        <div class="hero-description">
            Powered by Adaptive Instance Normalization (AdaIN)
        </div>

        <div class="divider"></div>
        """
    )


    with gr.Row():

        content_input = gr.Image(
            type="pil",
            label="📷 Content Image"
        )


        style_input = gr.Image(
            type="pil",
            label="🎨 Style Image"
        )


    strength = gr.Slider(
        minimum=0,
        maximum=1,
        value=0.8,
        step=0.1,
        label="✨ Style Strength"
    )


    generate_button = gr.Button(
        "🚀 Generate Artwork",
        variant="primary"
    )


    output_image = gr.Image(
        type="pil",
        label="🖼️ Generated Artwork"
    )


    generate_button.click(
        fn=stylize,
        inputs=[
            content_input,
            style_input,
            strength
        ],
        outputs=output_image
    )


# ==========================================
# Launch Application
# ==========================================

if __name__ == "__main__":

    demo.launch()
