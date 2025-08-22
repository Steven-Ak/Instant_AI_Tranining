import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO
import base64


st.set_page_config(
    page_title="Stable Diffusion App", 
    page_icon="🎨", 
    layout="centered"
)


st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h1 style="color:#4CAF50;">🎨 Stable Diffusion</h1>
        <p style="color:gray; font-size:18px;">
            Generate stunning AI images from your imagination ✨
        </p>
    </div>
    <hr style="border:1px solid #eee; margin: 15px 0;">
    """,
    unsafe_allow_html=True
)


st.markdown("### 📝 Enter Your Prompt")
prompt = st.text_area(
    "Describe the image you want to create:",
    "A futuristic city in the clouds",
    height=100,
    label_visibility="collapsed"
)


with st.sidebar:
    st.header("⚙️ Generation Settings")
    steps = st.slider("🔄 Steps (quality vs speed)", 10, 50, 25)
    guidance = st.slider("🎯 Guidance scale", 1.0, 15.0, 7.5)
    size = st.selectbox("📐 Image size", ["512x512"])  # safe for 6GB VRAM
    width, height = map(int, size.split("x"))


@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()
    return pipe

pipe = load_model()


st.markdown(
    "<div style='text-align:center; margin: 20px;'>",
    unsafe_allow_html=True
)
generate_btn = st.button("🚀 Generate Image", type="primary")
st.markdown("</div>", unsafe_allow_html=True)


if generate_btn and prompt.strip() != "":
    with st.spinner("✨ Crafting your masterpiece... please wait."):
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height
        ).images[0]

        # Show result nicely
        st.markdown("### 🖼️ Generated Image")
        st.image(image, use_container_width=True, caption="Your AI Creation")

        # Download button (styled)
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f"""
            <div style="text-align:center; margin-top: 15px;">
                <a href="data:file/png;base64,{b64}" download="generated.png"
                   style="display:inline-block; padding:12px 25px; 
                          background:#4CAF50; color:white; 
                          border-radius:8px; text-decoration:none; font-size:16px;">
                   📥 Download Image
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
elif generate_btn:
    st.warning("⚠️ Please enter a prompt before generating.")