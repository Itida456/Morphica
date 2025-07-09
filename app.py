import streamlit as st
import boto3
import json
import base64
from PIL import Image
import io
import random
from botocore.exceptions import ClientError
import base64
import os

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


st.set_page_config(page_title="AI Image Generator", layout="centered")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'prompt_memory' not in st.session_state:
    st.session_state.prompt_memory = []

# --- Helper functions ---
def create_boto3_client(service_name, region):
    return boto3.client(service_name, region_name=region)

def process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    valid_sizes = [512, 576, 640, 704, 768, 832, 896, 960, 1024]
    aspect_ratio = image.width / image.height
    if aspect_ratio >= 1:
        width = min(valid_sizes, key=lambda x: abs(x - image.width))
        height = min(valid_sizes, key=lambda x: abs(x - (width / aspect_ratio)))
    else:
        height = min(valid_sizes, key=lambda x: abs(x - image.height))
        width = min(valid_sizes, key=lambda x: abs(x - (height * aspect_ratio)))
    width = max((width // 64) * 64, 512)
    height = max((height // 64) * 64, 512)
    return image.resize((width, height), Image.Resampling.LANCZOS)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode()

def generate_text_to_image(prompt, model_id, region):
    client = create_boto3_client("bedrock-runtime", region)
    if "stability" in model_id.lower():
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 10,
            "steps": 30,
            "seed": random.randint(0, 4294967295),
            "width": 1024,
            "height": 1024
        }
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json"
        )
        return json.loads(response["body"].read())["artifacts"][0]["base64"]
    elif "titan" in model_id.lower():
        payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "premium",
                "height": 1024,
                "width": 1024,
                "cfgScale": 10.0,
                "seed": random.randint(0, 2147483647)
            }
        }
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json"
        )
        return json.loads(response["body"].read())["images"][0]

def generate_image_to_image(prompt, input_image, model_id, region):
    client = create_boto3_client("bedrock-runtime", region)
    processed_image = process_image(input_image)
    image_base64 = image_to_base64(processed_image)
    if "stability" in model_id.lower():
        payload = {
            "text_prompts": [{"text": prompt}],
            "init_image": image_base64,
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.5,
            "cfg_scale": 10,
            "steps": 30,
            "seed": random.randint(0, 4294967295)
        }
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json"
        )
        return json.loads(response["body"].read())["artifacts"][0]["base64"]
    elif "titan" in model_id.lower():
        payload = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": prompt,
                "images": [image_base64]
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "premium",
                "cfgScale": 10.0,
                "seed": random.randint(0, 2147483647)
            }
        }
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json"
        )
        return json.loads(response["body"].read())["images"][0]

# --- Sidebar ---
st.sidebar.title("Settings")
region = st.sidebar.selectbox("AWS Region:", ["us-east-1", "us-west-2", "eu-west-1"])
st.session_state.region = region

model_options = {
    "Amazon Titan V1": "amazon.titan-image-generator-v1",
    "Stability AI SDXL": "stability.stable-diffusion-xl-v1"
}
st.sidebar.markdown("**Model Selection**")
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    list(model_options.keys()),
    help="""
- **Amazon Titan V1**: Basic version of Titan, still powerful.
- **Stability AI SDXL**: High creativity, strong textures, very detailed.
"""
)

model_id = model_options[selected_model]

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigate:")
if st.sidebar.button("Home", use_container_width=True):
    st.session_state.page = "Home"
if st.sidebar.button("Text to Image", use_container_width=True):
    st.session_state.page = "Text to Image"
if st.sidebar.button("Image to Image", use_container_width=True):
    st.session_state.page = "Image to Image"

with st.sidebar.expander(" Prompt History", expanded=True):
    if st.session_state.prompt_memory:
        for prompt in reversed(st.session_state.prompt_memory):
            st.markdown(f"- {prompt}")
        if st.button("Clear Prompt History"):
            st.session_state.prompt_memory = []
            st.rerun()
    else:
        st.write("No prompt history yet.")

# --- HOME PAGE ---
if st.session_state.page == "Home":
    def load_image_as_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    image_base64_list = [
        load_image_as_base64("public/ai-image-generator-hero-image.png"),
        load_image_as_base64("public/Anime-Boys-PNG-HD-Quality.png"),
        load_image_as_base64("public/elephant-hd-quality.png"),
        load_image_as_base64("public/pendant-lamp-over-river.png"),
    ]

    st.markdown(f"""
        <style>
            @keyframes fade {{
                0% {{opacity: 0;}}
                10% {{opacity: 1;}}
                25% {{opacity: 1;}}
                35% {{opacity: 0;}}
                100% {{opacity: 0;}}
            }}
            .carousel-container {{
                position: relative;
                width: 100%;
                max-width: 600px;
                height: 400px;
                margin: auto;
                overflow: hidden;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            }}
            .carousel-slide {{
                position: absolute;
                width: 100%;
                height: 100%;
                object-fit: cover;
                opacity: 0;
                animation-name: fade;
                animation-duration: 16s;
                animation-iteration-count: infinite;
            }}
            .carousel-slide:nth-child(1) {{ animation-delay: 0s; }}
            .carousel-slide:nth-child(2) {{ animation-delay: 4s; }}
            .carousel-slide:nth-child(3) {{ animation-delay: 8s; }}
            .carousel-slide:nth-child(4) {{ animation-delay: 12s; }}

            .center-buttons {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 30px;
            }}

            .custom-btn {{
                font-size: 1.2em;
                background-color: #ff6f61;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
            }}

            .custom-btn:hover {{
                background-color: #e85b50;
            }}

            .stApp [data-testid="collapsedControl"] {{
                display: block;
            }}
        </style>

        <h1 style='text-align: center; font-size: 3em;'>Create Images in <span style='color: #FF6F61;'>Seconds</span></h1>
        <p style='text-align: center; font-size: 1.2em;'>Use the power of Generative AI to turn text and photos into stunning visuals. Choose a model, select a style, and let the magic happen—instantly.</p>
        <br>

        <div class="carousel-container">
            <img src="data:image/png;base64,{image_base64_list[0]}" class="carousel-slide">
            <img src="data:image/png;base64,{image_base64_list[1]}" class="carousel-slide">
            <img src="data:image/png;base64,{image_base64_list[2]}" class="carousel-slide">
            <img src="data:image/png;base64,{image_base64_list[3]}" class="carousel-slide">
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Try Text-to-Image", use_container_width=True):
            st.session_state.page = "Text to Image"
            st.rerun()
    with col2:
        if st.button("Try Image-to-Image", use_container_width=True):
            st.session_state.page = "Image to Image"
            st.rerun()

    st.markdown("""
        <script>
        setTimeout(function() {
            const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.transform = 'translateX(-100%)';
            }
        }, 500);
        </script>
    """, unsafe_allow_html=True)


# "Text to Image"
elif st.session_state.page == "Text to Image":
    st.markdown("""
    <h2 style='text-align: center; font-size: 2.5em; color: #FF6F61; margin-bottom: 10px;'>
        Create Images Just by a Simple Text 
    </h2>
""", unsafe_allow_html=True)

    st.markdown("Generate images from text descriptions with style presets")

    style_presets = {
        "None": "",
        "Anime": "anime portrait of a character, beautiful lighting, soft shading, colorful background, high detail",
        "Portrait": "professional portrait, studio lighting, sharp focus, detailed",
        "Photorealistic": "photorealistic, ultra high resolution, sharp focus, professional photography",
        "Digital Art": "highly detailed digital painting, creative lighting, stylized illustration, vibrant colors",
        "Cinematic": "cinematic lighting, dramatic atmosphere, film photography style",
        "Dreamscape": "dreamy ethereal style, soft magical lighting, pastel dream clouds, surrealism art",
        "Cyberpunk": "cyberpunk style, neon lights, futuristic city, high contrast",
        "Sketch": "pencil sketch, hand-drawn lines, minimal shading, black and white",
        "Cartoon": "cartoon style illustration, bold lines, flat colors, cheerful"
    }

    style = st.selectbox("Style Preset", list(style_presets.keys()))
    prompt_input = st.text_area("Your Prompt", value="high quality, detailed, professional", height=100)

    final_prompt = f"{style_presets[style]}, {prompt_input}" if style != "None" else prompt_input

    st.markdown("** Final Prompt:**")
    st.info(final_prompt)

    col1, col2 = st.columns([2, 1])
    with col1:
        generate_btn = st.button("Generate Image", use_container_width=True)
    with col2:
        clear_btn = st.button(" Clear Prompt", use_container_width=True)

    if generate_btn:
        if not prompt_input.strip():
            st.error("Please enter a prompt")
        else:
            st.session_state.prompt_memory.append(final_prompt)
            try:
                with st.spinner("Let the model cook..."):
                    image_base64 = generate_text_to_image(final_prompt, model_id, region)
                    generated_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
                    st.image(generated_image, caption=" Generated Image")

                    img_buffer = io.BytesIO()
                    generated_image.save(img_buffer, format='PNG')
                    st.download_button("Download", img_buffer.getvalue(), file_name="generated_image.png", mime="image/png")

            except ClientError as e:
                st.error(f"Error: {e.response['Error']['Message']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if clear_btn:
        prompt_input = ""
        st.rerun()

# ========== IMAGE TO IMAGE PAGE ==========
elif st.session_state.page == "Image to Image":
    st.markdown("""
    <h2 style='text-align: center; font-size: 2.5em; color: #0ea3c4; margin-bottom: 10px;'>
        Create Images Just by Uploading Images 
    </h2>
""", unsafe_allow_html=True)

    st.markdown("Upload a photo and apply AI styles")

    style_options = {
        "None": "",
        "Anime": "anime style",
        "Portrait": "portrait style",
        "Photorealistic": "photorealistic",
        "Digital Art": "digital painting style",
        "Cinematic": "cinematic lighting",
        "Dreamscape": "dreamy ethereal style",
        "Cyberpunk": "cyberpunk aesthetic",
        "Sketch": "pencil sketch style",
        "Cartoon": "cartoon style"
    }

    uploaded_file = st.file_uploader("Upload your image:", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_column_width=True)

        style = st.selectbox("🖌️ Choose Style", list(style_options.keys()))
        final_prompt = style_options[style] if style != "None" else "enhance the image"

        if st.button("Transform Image", type="primary"):
            st.session_state.prompt_memory.append(final_prompt)
            try:
                with st.spinner("Transforming your image..."):
                    image_base64 = generate_image_to_image(final_prompt, input_image, model_id, region)
                    output_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(input_image, caption="Original", use_column_width=True)
                    with col2:
                        st.image(output_image, caption="Transformed", use_column_width=True)

                    img_buffer = io.BytesIO()
                    output_image.save(img_buffer, format='PNG')
                    st.download_button("Download", img_buffer.getvalue(), file_name="transformed_image.png", mime="image/png")

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload an image to get started.")
