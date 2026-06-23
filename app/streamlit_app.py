"""
Streamlit Web Application for Social Media Disinformation Detection
Interactive interface for uploading images and text to check credibility.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from configs.config import config
from models.image_branch import ImageBranch, ImageBranchWithELA
from models.text_branch import TextBranch
from utils.preprocessing import clean_text, sanitize_input
from utils.ela import compute_ela
from utils.metrics import compute_metrics


@st.cache_resource
def load_models():
    """Load trained models (cached for performance)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_model = ImageBranchWithELA(num_classes=2, pretrained=False).to(device)
    text_model = TextBranch(
        model_name=config.text.model_name,
        num_classes=2
    ).to(device)
    
    models_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
    
    image_checkpoint = models_dir / "image_branch_best.pth"
    text_checkpoint = models_dir / "text_branch_best.pth"
    
    if image_checkpoint.exists():
        image_model.load_state_dict(torch.load(image_checkpoint, map_location=device, weights_only=True))
        image_model.eval()
    else:
        st.warning("Image model checkpoint not found. Using untrained model.")
    
    if text_checkpoint.exists():
        text_model.load_state_dict(torch.load(text_checkpoint, map_location=device, weights_only=True))
        text_model.eval()
    else:
        st.warning("Text model checkpoint not found. Using untrained model.")
    
    return image_model, text_model, device


def predict_image(image: Image.Image, model, device):
    """Predict image authenticity."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    ela_image = compute_ela(image)
    
    original_tensor = transform(image).unsqueeze(0).to(device)
    ela_tensor = transform(ela_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(original_tensor, ela_tensor)
        probs = torch.softmax(output, dim=-1)
    
    return {
        "prediction": "Fake" if probs[0][1] > 0.5 else "Real",
        "confidence": float(probs[0][1]) if probs[0][1] > 0.5 else float(probs[0][0]),
        "fake_probability": float(probs[0][1]),
        "ela_image": ela_image
    }


def predict_text(text: str, model, tokenizer, device):
    """Predict text credibility."""
    cleaned_text = clean_text(text)
    
    encoding = tokenizer(
        cleaned_text,
        max_length=config.text.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.softmax(output, dim=-1)
    
    return {
        "prediction": "Fake" if probs[0][1] > 0.5 else "Real",
        "confidence": float(probs[0][1]) if probs[0][1] > 0.5 else float(probs[0][0]),
        "fake_probability": float(probs[0][1]),
        "cleaned_text": cleaned_text
    }


def combine_predictions(
    image_result: dict,
    text_result: dict,
    strategy: str = "average"
) -> dict:
    """Combine predictions from both branches."""
    img_prob = image_result["fake_probability"]
    txt_prob = text_result["fake_probability"]
    
    if strategy == "average":
        final_prob = (img_prob + txt_prob) / 2
    elif strategy == "max":
        final_prob = max(img_prob, txt_prob)
    elif strategy == "min":
        final_prob = min(img_prob, txt_prob)
    elif strategy == "weighted":
        final_prob = 0.4 * img_prob + 0.6 * txt_prob
    else:
        final_prob = (img_prob + txt_prob) / 2
    
    is_fake = final_prob > 0.5
    confidence = final_prob if is_fake else (1 - final_prob)
    
    return {
        "final_prediction": "Fake" if is_fake else "Real",
        "final_confidence": float(confidence),
        "final_fake_probability": float(final_prob),
        "image_contribution": float(img_prob),
        "text_contribution": float(txt_prob)
    }


def main():
    st.set_page_config(
        page_title="Disinformation Detection System",
        page_icon=" ",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .fake-result {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .real-result {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Social Media Disinformation Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dual-Branch Image and Text Credibility Classifier</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Configuration")
        
        fusion_strategy = st.selectbox(
            "Fusion Strategy",
            ["average", "weighted", "max", "min"],
            index=0,
            help="Method to combine image and text predictions"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system detects disinformation by analyzing both images and text
        using a dual-branch neural network architecture.
        
        **Image Branch:** ResNet-18 with Error Level Analysis
        **Text Branch:** DistilBERT transformer
        """)
        
        st.markdown("---")
        st.markdown("### API Status")
        st.info("  API coming soon!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image Analysis")
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a social media post image for analysis"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Text Analysis")
        text_input = st.text_area(
            "Enter post text",
            height=200,
            placeholder="Enter the text content of the social media post...",
            help="Paste the text content from a social media post"
        )
    
    st.markdown("---")
    
    if st.button("  Analyze Credibility", type="primary", use_container_width=True):
        if not uploaded_image and not text_input:
            st.error("Please upload an image or enter text to analyze.")
            return
        
        with st.spinner("Loading models..."):
            image_model, text_model, device = load_models()
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(config.text.model_name)
        
        col1, col2, col3 = st.columns(3)
        
        image_result = None
        text_result = None
        
        if uploaded_image:
            with col1:
                st.subheader("Image Result")
                with st.spinner("Analyzing image..."):
                    image_result = predict_image(image, image_model, device)
                
                if image_result["prediction"] == "Fake":
                    st.markdown('<div class="fake-result">', unsafe_allow_html=True)
                    st.error(f"  Image: **{image_result['prediction']}**")
                    st.markdown(f"Confidence: {image_result['confidence']:.1%}")
                    st.markdown(f"Fake Probability: {image_result['fake_probability']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="real-result">', unsafe_allow_html=True)
                    st.success(f"  Image: **{image_result['prediction']}**")
                    st.markdown(f"Confidence: {image_result['confidence']:.1%}")
                    st.markdown(f"Fake Probability: {image_result['fake_probability']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if "ela_image" in image_result:
                    st.subheader("Error Level Analysis")
                    st.image(image_result["ela_image"], caption="ELA Visualization", use_container_width=True)
        
        if text_input:
            with col2:
                st.subheader("Text Result")
                with st.spinner("Analyzing text..."):
                    text_result = predict_text(text_input, text_model, tokenizer, device)
                
                if text_result["prediction"] == "Fake":
                    st.markdown('<div class="fake-result">', unsafe_allow_html=True)
                    st.error(f"  Text: **{text_result['prediction']}**")
                    st.markdown(f"Confidence: {text_result['confidence']:.1%}")
                    st.markdown(f"Fake Probability: {text_result['fake_probability']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="real-result">', unsafe_allow_html=True)
                    st.success(f"  Text: **{text_result['prediction']}**")
                    st.markdown(f"Confidence: {text_result['confidence']:.1%}")
                    st.markdown(f"Fake Probability: {text_result['fake_probability']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        if image_result and text_result:
            with col3:
                st.subheader("Combined Result")
                combined = combine_predictions(image_result, text_result, fusion_strategy)
                
                if combined["final_prediction"] == "Fake":
                    st.markdown('<div class="fake-result">', unsafe_allow_html=True)
                    st.error(f"  Final: **{combined['final_prediction']}**")
                    st.markdown(f"Overall Confidence: {combined['final_confidence']:.1%}")
                    st.markdown(f"Fake Probability: {combined['final_fake_probability']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="real-result">', unsafe_allow_html=True)
                    st.success(f"  Final: **{combined['final_prediction']}**")
                    st.markdown(f"Overall Confidence: {combined['final_confidence']:.1%}")
                    st.markdown(f"Fake Probability: {combined['final_fake_probability']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("**Fusion Breakdown:**")
                st.metric("Image Contribution", f"{combined['image_contribution']:.1%}")
                st.metric("Text Contribution", f"{combined['text_contribution']:.1%}")
                st.metric("Final Score", f"{combined['final_fake_probability']:.1%}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Social Media Disinformation Detection System | Federal University of Technology Minna</p>
    <p>Department of Cyber Security Science | 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
