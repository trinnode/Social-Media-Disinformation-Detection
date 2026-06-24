"""
Social Media Disinformation Detection — Streamlit Application
Dual-Branch Image and Text Credibility Classifier
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from configs.config import config
from models.image_branch import ImageBranch, ImageBranchWithELA
from models.text_branch import TextBranch
from utils.preprocessing import clean_text, sanitize_input
from utils.ela import compute_ela

TEMPERATURE = 1.0
CONFIDENCE_THRESHOLD = 0.55

st.set_page_config(
    page_title="DisInfo Shield — Disinformation Detection",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = ImageBranchWithELA(num_classes=2, pretrained=False).to(device)
    text_model = TextBranch(model_name=config.text.model_name, num_classes=2).to(device)

    models_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
    img_ckpt = models_dir / "image_branch_best.pth"
    txt_ckpt = models_dir / "text_branch_best.pth"

    if img_ckpt.exists():
        image_model.load_state_dict(torch.load(img_ckpt, map_location=device, weights_only=True))
        image_model.eval()
    if txt_ckpt.exists():
        text_model.load_state_dict(torch.load(txt_ckpt, map_location=device, weights_only=True))
        text_model.eval()

    return image_model, text_model, device


def calibrate(logits, temperature=TEMPERATURE):
    scaled = logits / temperature
    return torch.softmax(scaled, dim=-1)


def predict_image(image: Image.Image, model, device):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if image.mode != "RGB":
        image = image.convert("RGB")

    ela_image = compute_ela(image)
    orig_tensor = transform(image).unsqueeze(0).to(device)
    ela_tensor = transform(ela_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(orig_tensor, ela_tensor)
        probs = calibrate(logits)

    fake_prob = float(probs[0][1])
    real_prob = float(probs[0][0])

    if fake_prob > real_prob and fake_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Fake", fake_prob
    elif real_prob > fake_prob and real_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Real", real_prob
    else:
        prediction, confidence = "Uncertain", max(fake_prob, real_prob)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "ela_image": ela_image,
    }


def predict_text(text: str, model, tokenizer, device):
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        max_length=config.text.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = calibrate(logits)

    fake_prob = float(probs[0][1])
    real_prob = float(probs[0][0])

    if fake_prob > real_prob and fake_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Fake", fake_prob
    elif real_prob > fake_prob and real_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Real", real_prob
    else:
        prediction, confidence = "Uncertain", max(fake_prob, real_prob)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "cleaned_text": cleaned,
    }


def combine_predictions(image_result, text_result, strategy="average"):
    img_prob = image_result["fake_probability"]
    txt_prob = text_result["fake_probability"]

    if strategy == "average":
        final = (img_prob + txt_prob) / 2
    elif strategy == "weighted":
        final = 0.4 * img_prob + 0.6 * txt_prob
    elif strategy == "max":
        final = max(img_prob, txt_prob)
    elif strategy == "min":
        final = min(img_prob, txt_prob)
    else:
        final = (img_prob + txt_prob) / 2

    if final >= 0.55:
        label, confidence = "Fake", final
    elif final <= 0.45:
        label, confidence = "Real", 1 - final
    else:
        label, confidence = "Uncertain", max(final, 1 - final)

    return {
        "final_prediction": label,
        "final_confidence": confidence,
        "final_fake_probability": final,
        "final_real_probability": 1 - final,
        "image_contribution": img_prob,
        "text_contribution": txt_prob,
    }


def init_session():
    defaults = {
        "history": [],
        "analysis_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def save_to_history(entry):
    st.session_state.history.insert(0, entry)
    st.session_state.analysis_count += 1


def render_inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0a0a0f;
    }

    .block-container {
        padding-top: 1rem !important;
        max-width: 1200px !important;
    }

    .hero-section {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-section::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(236,72,153,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a78bfa;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 0.75rem;
    }

    .input-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .input-card-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 0.75rem;
    }

    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        min-height: 180px;
    }
    .result-card-fake {
        border-color: rgba(239,68,68,0.3);
        background: rgba(239,68,68,0.05);
    }
    .result-card-real {
        border-color: rgba(34,197,94,0.3);
        background: rgba(34,197,94,0.05);
    }
    .result-card-uncertain {
        border-color: rgba(250,204,21,0.3);
        background: rgba(250,204,21,0.05);
    }

    .verdict-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 0.5rem;
    }
    .verdict-text {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }
    .verdict-fake { color: #ef4444; }
    .verdict-real { color: #22c55e; }
    .verdict-uncertain { color: #facc15; }
    .verdict-conf {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
        margin-top: 0.5rem;
    }

    .prob-bar-container {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 8px;
        margin: 0.75rem 0;
        overflow: hidden;
    }
    .prob-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    .prob-bar-fake { background: linear-gradient(90deg, #f97316, #ef4444); }
    .prob-bar-real { background: linear-gradient(90deg, #22c55e, #10b981); }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
    }
    .metric-value {
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(255,255,255,0.8);
    }

    .section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.3);
        margin: 1.5rem 0 0.75rem 0;
    }

    .history-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .history-time {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.3);
        margin-bottom: 0.25rem;
    }
    .history-verdict {
        font-size: 1rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }
    .history-detail {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        line-height: 1.5;
    }

    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .footer-section {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 3rem;
    }
    .footer-text {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.25);
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.3);
    }

    .stTextArea textarea,
    .stFileUploader {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 1px rgba(99,102,241,0.3) !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(10,10,20,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: rgba(255,255,255,0.9) !important;
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: rgba(255,255,255,0.7) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    init_session()
    render_inject_css()

    # ── HERO ──
    st.markdown("""
    <div class="hero-section">
        <div style="position:relative; z-index:1;">
            <div class="hero-badge">DUAL-BRANCH NEURAL NETWORK</div>
            <h1 class="hero-title">DisInfo Shield</h1>
            <p class="hero-subtitle">
                AI-powered detection of social media disinformation through
                combined image forensics and natural language analysis
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0 0.5rem 0;">
            <div style="font-size:1.5rem; font-weight:800;
                 background:linear-gradient(135deg,#a78bfa,#ec4899);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                 DISINFO SHIELD
            </div>
            <div style="font-size:0.7rem; color:rgba(255,255,255,0.35);
                 letter-spacing:1px; text-transform:uppercase;">
                Configuration Panel
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">FUSION STRATEGY</div>', unsafe_allow_html=True)
        fusion_strategy = st.selectbox(
            "Method",
            ["average", "weighted", "max", "min"],
            index=0,
            label_visibility="collapsed",
        )

        strategy_descriptions = {
            "average": "Equal weighting of both branches (50/50)",
            "weighted": "Text-weighted fusion (40% image, 60% text)",
            "max": "Takes the higher fake probability",
            "min": "Takes the lower fake probability",
        }
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.2);
             border-radius:8px; padding:0.6rem 0.8rem; margin-top:0.5rem;">
            <div style="font-size:0.7rem; color:#a78bfa; font-weight:600;">
                {strategy_descriptions[fusion_strategy]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">SYSTEM INFO</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-section">
            <div class="metric-row">
                <span class="metric-label">Image Branch</span>
                <span class="metric-value">ResNet-18 + ELA</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Text Branch</span>
                <span class="metric-value">DistilBERT</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Temperature</span>
                <span class="metric-value">1.0</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Confidence Threshold</span>
                <span class="metric-value">0.55</span>
            </div>
            <div class="metric-row" style="border:none;">
                <span class="metric-label">Device</span>
                <span class="metric-value">CPU</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">ANALYSIS HISTORY</div>', unsafe_allow_html=True)
        count = st.session_state.analysis_count
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="metric-row">
                <span class="metric-label">Total Analyses</span>
                <span class="metric-value">{count}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Session</span>
                <span class="metric-value">Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.history:
            for i, entry in enumerate(st.session_state.history[:10]):
                verdict_color = {"Fake": "#ef4444", "Real": "#22c55e", "Uncertain": "#facc15"}
                color = verdict_color.get(entry["verdict"], "#9ca3af")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);
                     border-radius:8px; padding:0.5rem 0.7rem; margin-bottom:0.4rem;
                     border-left:3px solid {color};">
                    <div style="font-size:0.65rem; color:rgba(255,255,255,0.3);">{entry['time']}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:{color};">{entry['verdict']}</div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.4);">{entry['summary']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── INPUT SECTION ──
    st.markdown('<div class="section-label">INPUT</div>', unsafe_allow_html=True)

    col_img, col_txt = st.columns(2)

    with col_img:
        st.markdown('<div class="input-card"><div class="input-card-label">IMAGE ANALYSIS</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Upload image for forensic analysis",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="img_uploader",
        )
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_txt:
        st.markdown('<div class="input-card"><div class="input-card-label">TEXT ANALYSIS</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Enter post text",
            height=200,
            placeholder="Paste the text content of the social media post here...",
            label_visibility="collapsed",
            key="txt_input",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ANALYZE BUTTON ──
    if st.button("  Analyze Credibility", use_container_width=True, key="analyze_btn"):
        if not uploaded_image and not text_input:
            st.error("Please upload an image or enter text to analyze.")
            return

        with st.spinner("Loading models..."):
            image_model, text_model, device = load_models()
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(config.text.model_name)

        image_result = None
        text_result = None
        has_image = uploaded_image is not None
        has_text = bool(text_input and text_input.strip())

        # ── RESULTS ──
        st.markdown('<div class="section-label">RESULTS</div>', unsafe_allow_html=True)

        if has_image and has_text:
            result_cols = st.columns(3)
        elif has_image:
            result_cols = st.columns(2)
        else:
            result_cols = st.columns(2)

        col_idx = 0

        if has_image:
            with result_cols[col_idx]:
                with st.spinner("Running image forensics..."):
                    image_result = predict_image(image, image_model, device)

                verdict = image_result["prediction"]
                fake_p = image_result["fake_probability"]
                conf = image_result["confidence"]

                card_class = {"Fake": "result-card-fake", "Real": "result-card-real", "Uncertain": "result-card-uncertain"}.get(verdict, "")
                verdict_class = {"Fake": "verdict-fake", "Real": "verdict-real", "Uncertain": "verdict-uncertain"}.get(verdict, "")

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="verdict-label">IMAGE VERDICT</div>
                    <div class="verdict-text {verdict_class}">{verdict}</div>
                    <div class="verdict-conf">Confidence: {conf:.1%}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar prob-bar-fake" style="width:{fake_p*100:.1f}%"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:rgba(255,255,255,0.4);">
                        <span>Fake: {fake_p:.1%}</span>
                        <span>Real: {image_result['real_probability']:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("  Error Level Analysis Visualization"):
                    st.image(image_result["ela_image"], caption="ELA Difference Map", use_container_width=True)

            col_idx += 1

        if has_text:
            with result_cols[col_idx]:
                with st.spinner("Running text analysis..."):
                    text_result = predict_text(text_input, text_model, tokenizer, device)

                verdict = text_result["prediction"]
                fake_p = text_result["fake_probability"]
                conf = text_result["confidence"]

                card_class = {"Fake": "result-card-fake", "Real": "result-card-real", "Uncertain": "result-card-uncertain"}.get(verdict, "")
                verdict_class = {"Fake": "verdict-fake", "Real": "verdict-real", "Uncertain": "verdict-uncertain"}.get(verdict, "")

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="verdict-label">TEXT VERDICT</div>
                    <div class="verdict-text {verdict_class}">{verdict}</div>
                    <div class="verdict-conf">Confidence: {conf:.1%}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar prob-bar-fake" style="width:{fake_p*100:.1f}%"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:rgba(255,255,255,0.4);">
                        <span>Fake: {fake_p:.1%}</span>
                        <span>Real: {text_result['real_probability']:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_idx += 1

        if image_result and text_result:
            combined = combine_predictions(image_result, text_result, fusion_strategy)

            with result_cols[col_idx]:
                verdict = combined["final_prediction"]
                fake_p = combined["final_fake_probability"]
                conf = combined["final_confidence"]

                card_class = {"Fake": "result-card-fake", "Real": "result-card-real", "Uncertain": "result-card-uncertain"}.get(verdict, "")
                verdict_class = {"Fake": "verdict-fake", "Real": "verdict-real", "Uncertain": "verdict-uncertain"}.get(verdict, "")

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="verdict-label">FUSED VERDICT</div>
                    <div class="verdict-text {verdict_class}">{verdict}</div>
                    <div class="verdict-conf">Confidence: {conf:.1%}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar prob-bar-fake" style="width:{fake_p*100:.1f}%"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:rgba(255,255,255,0.4);">
                        <span>Fake: {fake_p:.1%}</span>
                        <span>Real: {combined['final_real_probability']:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── FUSION BREAKDOWN ──
        if image_result and text_result:
            st.markdown('<div class="section-label">FUSION BREAKDOWN</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                 border-radius:12px; padding:1.5rem;">
                <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem;">
                    <div style="text-align:center;">
                        <div style="font-size:0.65rem; font-weight:700; letter-spacing:1px;
                             text-transform:uppercase; color:rgba(255,255,255,0.35); margin-bottom:0.5rem;">
                            IMAGE SCORE
                        </div>
                        <div style="font-size:1.5rem; font-weight:800; color:#a78bfa;">
                            {image_result['fake_probability']:.1%}
                        </div>
                        <div style="font-size:0.7rem; color:rgba(255,255,255,0.35);">fake probability</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:0.65rem; font-weight:700; letter-spacing:1px;
                             text-transform:uppercase; color:rgba(255,255,255,0.35); margin-bottom:0.5rem;">
                            TEXT SCORE
                        </div>
                        <div style="font-size:1.5rem; font-weight:800; color:#ec4899;">
                            {text_result['fake_probability']:.1%}
                        </div>
                        <div style="font-size:0.7rem; color:rgba(255,255,255,0.35);">fake probability</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:0.65rem; font-weight:700; letter-spacing:1px;
                             text-transform:uppercase; color:rgba(255,255,255,0.35); margin-bottom:0.5rem;">
                            STRATEGY
                        </div>
                        <div style="font-size:1.2rem; font-weight:700; color:rgba(255,255,255,0.7);
                             text-transform:uppercase;">
                            {fusion_strategy}
                        </div>
                        <div style="font-size:0.7rem; color:rgba(255,255,255,0.35);">fusion method</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:0.65rem; font-weight:700; letter-spacing:1px;
                             text-transform:uppercase; color:rgba(255,255,255,0.35); margin-bottom:0.5rem;">
                            FINAL SCORE
                        </div>
                        <div style="font-size:1.5rem; font-weight:800; color:#f97316;">
                            {fake_p:.1%}
                        </div>
                        <div style="font-size:0.7rem; color:rgba(255,255,255,0.35);">fake probability</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── SAVE TO HISTORY ──
            img_summary = f"Image: {image_result['prediction']} ({image_result['fake_probability']:.1%})"
            txt_summary = f"Text: {text_result['prediction']} ({text_result['fake_probability']:.1%})"
            fuse_summary = f"Fused: {combined['final_prediction']} ({combined['final_fake_probability']:.1%})"

            entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "verdict": combined["final_prediction"],
                "image_prob": image_result["fake_probability"],
                "text_prob": text_result["fake_probability"],
                "final_prob": combined["final_fake_probability"],
                "strategy": fusion_strategy,
                "summary": f"{img_summary} | {txt_summary} | {fuse_summary}",
                "text_preview": text_input[:80] if text_input else "(no text)",
            }
            save_to_history(entry)
            st.rerun()

        elif image_result:
            entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "verdict": image_result["prediction"],
                "image_prob": image_result["fake_probability"],
                "text_prob": None,
                "final_prob": image_result["fake_probability"],
                "strategy": "image-only",
                "summary": f"Image: {image_result['prediction']} ({image_result['fake_probability']:.1%})",
                "text_preview": "(no text)",
            }
            save_to_history(entry)
            st.rerun()

        elif text_result:
            entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "verdict": text_result["prediction"],
                "image_prob": None,
                "text_prob": text_result["fake_probability"],
                "final_prob": text_result["fake_probability"],
                "strategy": "text-only",
                "summary": f"Text: {text_result['prediction']} ({text_result['fake_probability']:.1%})",
                "text_preview": text_input[:80] if text_input else "(no text)",
            }
            save_to_history(entry)
            st.rerun()

    # ── ANALYSIS HISTORY (below results) ──
    if st.session_state.history:
        st.markdown('<div class="section-label">ANALYSIS HISTORY</div>', unsafe_allow_html=True)

        for i, entry in enumerate(st.session_state.history):
            verdict_color = {"Fake": "#ef4444", "Real": "#22c55e", "Uncertain": "#facc15"}.get(entry["verdict"], "#9ca3af")
            bg_color = {"Fake": "rgba(239,68,68,0.04)", "Real": "rgba(34,197,94,0.04)", "Uncertain": "rgba(250,204,21,0.04)"}.get(entry["verdict"], "rgba(255,255,255,0.02)")

            img_bar = ""
            if entry["image_prob"] is not None:
                img_bar = f"""
                <div style="flex:1;">
                    <div style="font-size:0.65rem; color:rgba(255,255,255,0.35); margin-bottom:0.25rem;">IMAGE</div>
                    <div class="prob-bar-container" style="margin:0;">
                        <div class="prob-bar prob-bar-fake" style="width:{entry['image_prob']*100:.1f}%"></div>
                    </div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); text-align:right;">
                        {entry['image_prob']:.1%}
                    </div>
                </div>
                """

            txt_bar = ""
            if entry["text_prob"] is not None:
                txt_bar = f"""
                <div style="flex:1;">
                    <div style="font-size:0.65rem; color:rgba(255,255,255,0.35); margin-bottom:0.25rem;">TEXT</div>
                    <div class="prob-bar-container" style="margin:0;">
                        <div class="prob-bar prob-bar-fake" style="width:{entry['text_prob']*100:.1f}%"></div>
                    </div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); text-align:right;">
                        {entry['text_prob']:.1%}
                    </div>
                </div>
                """

            st.markdown(f"""
            <div class="history-card" style="background:{bg_color}; border-left:3px solid {verdict_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="history-time">{entry['time']}</div>
                        <div class="history-verdict" style="color:{verdict_color};">
                            #{i+1} — {entry['verdict']}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.65rem; color:rgba(255,255,255,0.3);">FINAL SCORE</div>
                        <div style="font-size:1.2rem; font-weight:700; color:{verdict_color};">
                            {entry['final_prob']:.1%}
                        </div>
                    </div>
                </div>
                <div style="display:flex; gap:1rem; margin-top:0.75rem;">
                    {img_bar}
                    {txt_bar}
                </div>
                <div class="history-detail" style="margin-top:0.5rem;">
                    {entry['text_preview']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown("""
    <div class="footer-section">
        <div class="footer-text">
            DisInfo Shield — Social Media Disinformation Detection System<br>
            Department of Cyber Security Science — Federal University of Technology Minna — 2026
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
