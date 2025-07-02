import streamlit as st
import boto3
import base64
import json
from PIL import Image
from io import BytesIO

# Bedrock client setup
client = boto3.client("bedrock-runtime", region_name="us-east-1")

st.set_page_config(page_title="AI Profile Picture Generator", layout="centered")
st.title("🖼️ AI Profile Picture Generator (Image-to-Image via Titan)")

# UI inputs
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
style_prompt = st.text_input("Enter a style (e.g., 'anime', 'pixar', 'cyberpunk')", value="anime")

if st.button("Generate AI Profile Picture") and uploaded_file and style_prompt:
    with st.spinner("Transforming your image using Amazon Bedrock Titan..."):

        image_bytes = uploaded_file.read()
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": style_prompt,
                "image": encoded,
                "cfgScale": 8,
                "seed": 42,
                "steps": 50
            }
        }

        try:
            response = client.invoke_model(
                modelId="amazon.titan-image-generator-v1",
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )
            result = json.loads(response["body"].read())
            gen_img_b64 = result["images"][0]
            st.success("✅ AI Profile Picture Generated!")
            st.image(base64.b64decode(gen_img_b64), use_column_width=True)

        except Exception as e:
            st.error(f"❌ Failed to generate image: {e}")
