import streamlit as st
import boto3
import json
import base64
from PIL import Image
import io
import random
from botocore.exceptions import ClientError

st.set_page_config(page_title="AI Image Generator", layout="centered")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Text to Image"

def create_boto3_client(service_name, region):
    return boto3.client(service_name, region_name=region)

def process_image(image):
    """Process image to meet model requirements"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to valid dimensions (multiples of 64)
    valid_sizes = [512, 576, 640, 704, 768, 832, 896, 960, 1024]
    
    # Find best fit maintaining aspect ratio
    aspect_ratio = image.width / image.height
    
    if aspect_ratio >= 1:  # Landscape or square
        width = min(valid_sizes, key=lambda x: abs(x - image.width))
        height = min(valid_sizes, key=lambda x: abs(x - (width / aspect_ratio)))
    else:  # Portrait
        height = min(valid_sizes, key=lambda x: abs(x - image.height))
        width = min(valid_sizes, key=lambda x: abs(x - (height * aspect_ratio)))
    
    # Ensure dimensions are multiples of 64
    width = (width // 64) * 64
    height = (height // 64) * 64
    
    # Minimum size check
    width = max(width, 512)
    height = max(height, 512)
    
    return image.resize((width, height), Image.Resampling.LANCZOS)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode()

def generate_text_to_image(prompt, model_id, region):
    """Generate image from text only"""
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
        response_json = json.loads(response["body"].read())
        return response_json["artifacts"][0]["base64"]
    
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
        response_json = json.loads(response["body"].read())
        return response_json["images"][0]
    
    else:  # Default to stability format
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 10,
            "steps": 30,
            "width": 1024,
            "height": 1024
        }
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json"
        )
        response_json = json.loads(response["body"].read())
        return response_json["artifacts"][0]["base64"]

def generate_image_to_image(prompt, input_image, model_id, region):
    """Transform given image based on style"""
    client = create_boto3_client("bedrock-runtime", region)
    
    # Process image to meet requirements
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
        
        response_json = json.loads(response["body"].read())
        return response_json["artifacts"][0]["base64"]
    
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
        
        response_json = json.loads(response["body"].read())
        return response_json["images"][0]

# Sidebar
st.sidebar.title("Settings")
region = st.sidebar.selectbox("AWS Region:", ["us-east-1", "us-west-2", "eu-west-1"])
st.session_state.region = region

# Updated model selection with newer models
model_options = {
    "Amazon Titan V2": "amazon.titan-image-generator-v2",
    "Amazon Titan V1": "amazon.titan-image-generator-v1",
    "Stability AI SDXL": "stability.stable-diffusion-xl-v1",
    "Stability AI SD3": "stability.sd3-medium-v1"
}
selected_model = st.sidebar.selectbox("Model:", list(model_options.keys()))
model_id = model_options[selected_model]

# Page Navigation
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
if st.sidebar.button("Text to Image", use_container_width=True):
    st.session_state.page = "Text to Image"
if st.sidebar.button("Image to Image", use_container_width=True):
    st.session_state.page = "Image to Image"

# Main Content
if st.session_state.page == "Text to Image":
    st.title("🎨 Text to Image Generator")
    st.markdown("Generate images from text descriptions")
    
    # Text to Image style presets
    text_to_image_styles = {
        "None": "",
        "Anime": "anime portrait of a character, beautiful lighting, soft shading, colorful background, high detail",
        "Portrait": "professional portrait, studio lighting, sharp focus, detailed",
        "Photorealistic": "photorealistic, ultra high resolution, sharp focus, professional photography",
        "Digital Art": "highly detailed digital painting, creative lighting, stylized illustration, vibrant colors",
        "Cinematic": "cinematic lighting, dramatic atmosphere, film photography style",
        "Dreamscape": "in dreamy ethereal style, soft magical lighting, keep original subject, pastel dream clouds, surrealism art",
        "Cyberpunk": "cyberpunk style, neon lights, futuristic city, high contrast",
        "Sketch": "pencil sketch, hand-drawn lines, minimal shading, black and white",
        "Cartoon": "cartoon style illustration, bold lines, flat colors, cheerful"
    }
    
    style = st.selectbox("Style Preset:", list(text_to_image_styles.keys()))
    prompt_input = st.text_area("Describe your image:", height=100, 
                                value="high quality, detailed, professional")
    
    # Combine style and prompt
    if style != "None":
        final_prompt = f"{text_to_image_styles[style]}, {prompt_input}"
    else:
        final_prompt = prompt_input
    
    st.markdown("**Final Prompt Preview:**")
    st.info(final_prompt)
    
    if st.button("Generate Image", type="primary"):
        if not prompt_input.strip():
            st.error("Please enter a prompt")
        else:
            try:
                with st.spinner("Generating your image..."):
                    image_base64 = generate_text_to_image(final_prompt, model_id, region)
                    
                    # Display result
                    generated_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
                    st.image(generated_image, caption="Generated Image")
                    
                    # Download
                    img_buffer = io.BytesIO()
                    generated_image.save(img_buffer, format='PNG')
                    st.download_button(
                        "Download Image",
                        data=img_buffer.getvalue(),
                        file_name=f"text_to_image_{random.randint(1000,9999)}.png",
                        mime="image/png"
                    )
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                
                st.error(f"Error: {error_message}")
                
                if "ValidationException" in error_code:
                    st.info("Try: Enable model access in AWS Bedrock Console")
                elif "AccessDenied" in error_code:
                    st.info("Try: Check AWS credentials and permissions")
                elif "ThrottlingException" in error_code:
                    st.info("Try: Wait a moment and try again")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try: Check model availability in your region")

elif st.session_state.page == "Image to Image":
    st.title("🖼️ Image to Image Generator")
    st.markdown("Transform your images with AI styles")
    
    # Image to Image style presets
    image_to_image_styles = {
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
    
    # Image upload
    uploaded_file = st.file_uploader("Upload your image:", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.markdown("**Image Info:**")
            st.write(f"Size: {uploaded_image.size[0]} x {uploaded_image.size[1]}")
            st.write(f"Format: {uploaded_image.format}")
            st.write(f"Mode: {uploaded_image.mode}")
        
        style = st.selectbox("Style Transformation:", list(image_to_image_styles.keys()))
        
        # Use only the style preset
        final_prompt = image_to_image_styles[style] if style != "None" else "enhance the image"
        
        if style != "None":
            st.markdown("**Selected Style:**")
            st.info(f"✨ {style} - {image_to_image_styles[style]}")
        else:
            st.markdown("**Selected Style:**")
            st.info("🔧 Basic Enhancement - enhance the image")
        
        if st.button("Transform Image", type="primary"):
            try:
                with st.spinner("Transforming your image..."):
                    image_base64 = generate_image_to_image(final_prompt, uploaded_image, model_id, region)
                    
                    # Display result
                    generated_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(uploaded_image, caption="Original", use_column_width=True)
                    with col2:
                        st.image(generated_image, caption="Transformed", use_column_width=True)
                    
                    # Download
                    img_buffer = io.BytesIO()
                    generated_image.save(img_buffer, format='PNG')
                    st.download_button(
                        "Download Transformed Image",
                        data=img_buffer.getvalue(),
                        file_name=f"image_to_image_{random.randint(1000,9999)}.png",
                        mime="image/png"
                    )
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                
                st.error(f"Error: {error_message}")
                
                if "ValidationException" in error_code:
                    st.info("Try: Enable model access in AWS Bedrock Console")
                elif "AccessDenied" in error_code:
                    st.info("Try: Check AWS credentials and permissions")
                elif "ThrottlingException" in error_code:
                    st.info("Try: Wait a moment and try again")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try: Check model availability in your region")
    
    else:
        st.info("👆 Please upload an image to get started")
        st.markdown("### Supported formats:")
        st.markdown("- PNG, JPG, JPEG")
        st.markdown("- Max file size: 200MB")
        st.markdown("- Recommended: Square images (1024x1024) for best results")