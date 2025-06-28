import streamlit as st
import boto3
import base64
from PIL import Image
import io

# Set up Streamlit
st.title("AI Profile Picture Generator 🎨")

uploaded_image = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Enter a style (e.g., 'anime', 'pixar style', 'cyberpunk')")
generate = st.button("Generate AI Profile Picture")

if generate and uploaded_image and prompt:
    # Convert the uploaded image to base64
    image_bytes = uploaded_image.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    st.info("Sending request to Bedrock...")

    # Connect to AWS Bedrock
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Payload for Titan Image Generator G1 (or Stable Diffusion)
    payload = {
        "taskType": "IN_PAINTING",  # Use IN_PAINTING or TEXT_IMAGE
        "image": base64_image,
        "text": prompt,
        "cfgScale": 8,
        "steps": 50,
        "seed": 42
    }

    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-image-generator-v1",
            body=bytes(str(payload), 'utf-8'),
            accept="application/json",
            contentType="application/json"
        )

        # Extract and decode the image
        response_body = response["body"].read()
        response_json = eval(response_body.decode("utf-8"))
        generated_image_base64 = response_json["images"][0]

        # Convert back to image
        generated_image = Image.open(io.BytesIO(base64.b64decode(generated_image_base64)))

        st.success("AI Profile Picture Generated!")
        st.image(generated_image, caption="Generated AI Photo", use_column_width=True)

    except Exception as e:
        st.error(f"Failed to generate image: {e}")
