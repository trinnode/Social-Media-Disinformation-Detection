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
    import time

    logs = []
    logs.append(("STEP 1: Input Received", f"Image mode: {image.mode}, Size: {image.size[0]}x{image.size[1]} pixels"))
    logs.append(("STEP 1: Input Received", f"Total pixels: {image.size[0] * image.size[1]:,}"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if image.mode != "RGB":
        logs.append(("STEP 2: Mode Conversion", f"Converted from {image.mode} to RGB"))
        image = image.convert("RGB")
    else:
        logs.append(("STEP 2: Mode Conversion", "Already RGB, no conversion needed"))

    logs.append(("STEP 3: Error Level Analysis", "Recompressing image at JPEG quality = 95..."))
    t0 = time.time()
    ela_image = compute_ela(image)
    ela_time = time.time() - t0

    ela_array = np.array(ela_image, dtype=np.float64)
    ela_mean = float(ela_array.mean())
    ela_std = float(ela_array.std())
    ela_max = float(ela_array.max())
    logs.append(("STEP 3: Error Level Analysis", f"ELA computed in {ela_time*1000:.1f}ms"))
    logs.append(("STEP 3: ELA Statistics", f"Mean error: {ela_mean:.2f} | Std dev: {ela_std:.2f} | Max error: {ela_max:.2f}"))
    logs.append(("STEP 3: ELA Interpretation", f"Lower mean ({ela_mean:.1f}) suggests consistent compression = likely authentic" if ela_mean < 15 else f"Higher mean ({ela_mean:.1f}) suggests compression anomalies = possible manipulation"))

    logs.append(("STEP 4: Image Transform", "Resizing to 224x224, converting to tensor, normalising with ImageNet stats"))
    logs.append(("STEP 4: Normalisation", "Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]"))
    orig_tensor = transform(image).unsqueeze(0).to(device)
    ela_tensor = transform(ela_image).unsqueeze(0).to(device)
    logs.append(("STEP 4: Tensor Shape", f"Original: {list(orig_tensor.shape)} | ELA: {list(ela_tensor.shape)}"))

    logs.append(("STEP 5: ResNet-18 Forward Pass", "Running dual-backbone inference (original + ELA branches)..."))
    t0 = time.time()
    with torch.no_grad():
        logits = model(orig_tensor, ela_tensor)
        probs = calibrate(logits)
    inf_time = time.time() - t0

    logit_real = float(logits[0][0])
    logit_fake = float(logits[0][1])
    logs.append(("STEP 5: Raw Logits", f"Logit[Real] = {logit_real:.4f} | Logit[Fake] = {logit_fake:.4f}"))
    logs.append(("STEP 5: Softmax", f"After softmax: P(Real) = {float(probs[0][0]):.4f} | P(Fake) = {float(probs[0][1]):.4f}"))
    logs.append(("STEP 5: Decision Threshold", f"Threshold: >= {CONFIDENCE_THRESHOLD} for confident, else Uncertain"))

    fake_prob = float(probs[0][1])
    real_prob = float(probs[0][0])

    if fake_prob > real_prob and fake_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Fake", fake_prob
    elif real_prob > fake_prob and real_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Real", real_prob
    else:
        prediction, confidence = "Uncertain", max(fake_prob, real_prob)

    logs.append(("STEP 5: Classification", f"Fake ({fake_prob:.4f}) {'>' if fake_prob > real_prob else '<'} Real ({real_prob:.4f})"))
    logs.append(("STEP 6: Final Verdict", f"{prediction} with confidence {confidence:.1%} | Inference time: {inf_time*1000:.1f}ms"))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "ela_image": ela_image,
        "logs": logs,
        "input_size": f"{orig_tensor.shape}",
        "inference_time_ms": inf_time * 1000,
    }


def predict_text(text: str, model, tokenizer, device):
    import time

    logs = []
    logs.append(("STEP 1: Input Received", f"Length: {len(text)} characters, {len(text.split())} words"))
    logs.append(("STEP 1: Raw Text", text[:120] + ("..." if len(text) > 120 else "")))

    cleaned = clean_text(text)
    logs.append(("STEP 2: Text Cleaning", "Removed URLs, @mentions, #hashtags, special characters, normalised whitespace"))
    logs.append(("STEP 2: Cleaned Text", cleaned[:120] + ("..." if len(cleaned) > 120 else "")))
    logs.append(("STEP 2: Cleaning Stats", f"Before: {len(text)} chars | After: {len(cleaned)} chars | Removed: {len(text) - len(cleaned)} chars"))

    logs.append(("STEP 3: Tokenisation", f"Tokenizer: DistilBERT WordPiece | Max length: {config.text.max_length} tokens"))
    t0 = time.time()
    encoding = tokenizer(
        cleaned,
        max_length=config.text.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tok_time = time.time() - t0

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    actual_tokens = int(attention_mask.sum().item())
    logs.append(("STEP 3: Tokenisation", f"Tokenised in {tok_time*1000:.1f}ms"))
    logs.append(("STEP 3: Token Stats", f"Total tokens: {input_ids.shape[1]} | Actual tokens: {actual_tokens} | Padding: {input_ids.shape[1] - actual_tokens} tokens"))
    logs.append(("STEP 3: Sample Tokens", f"First 10 token IDs: {input_ids[0][:10].tolist()}"))

    logs.append(("STEP 4: DistilBERT Forward Pass", "Running 6-layer transformer encoder with 12 attention heads..."))
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = calibrate(logits)
    inf_time = time.time() - t0

    logit_real = float(logits[0][0])
    logit_fake = float(logits[0][1])
    logs.append(("STEP 4: Raw Logits", f"Logit[Real] = {logit_real:.4f} | Logit[Fake] = {logit_fake:.4f}"))
    logs.append(("STEP 4: Softmax", f"After softmax: P(Real) = {float(probs[0][0]):.4f} | P(Fake) = {float(probs[0][1]):.4f}"))
    logs.append(("STEP 4: Decision Threshold", f"Threshold: >= {CONFIDENCE_THRESHOLD} for confident, else Uncertain"))

    fake_prob = float(probs[0][1])
    real_prob = float(probs[0][0])

    if fake_prob > real_prob and fake_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Fake", fake_prob
    elif real_prob > fake_prob and real_prob >= CONFIDENCE_THRESHOLD:
        prediction, confidence = "Real", real_prob
    else:
        prediction, confidence = "Uncertain", max(fake_prob, real_prob)

    logs.append(("STEP 4: Classification", f"Fake ({fake_prob:.4f}) {'>' if fake_prob > real_prob else '<'} Real ({real_prob:.4f})"))
    logs.append(("STEP 5: Final Verdict", f"{prediction} with confidence {confidence:.1%} | Inference time: {inf_time*1000:.1f}ms"))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "cleaned_text": cleaned,
        "logs": logs,
        "token_count": actual_tokens,
        "inference_time_ms": inf_time * 1000,
    }


def combine_predictions(image_result, text_result, strategy="average"):
    img_prob = image_result["fake_probability"]
    txt_prob = text_result["fake_probability"]

    logs = []
    logs.append(("STEP 1: Branch Outputs", f"Image P(Fake) = {img_prob:.4f} | Text P(Fake) = {txt_prob:.4f}"))
    logs.append(("STEP 1: Disagreement", f"Absolute difference = |{img_prob:.4f} - {txt_prob:.4f}| = {abs(img_prob - txt_prob):.4f}"))

    strategy_formulas = {
        "average": f"({img_prob:.4f} + {txt_prob:.4f}) / 2 = {((img_prob + txt_prob) / 2):.4f}",
        "weighted": f"0.4 * {img_prob:.4f} + 0.6 * {txt_prob:.4f} = {(0.4 * img_prob + 0.6 * txt_prob):.4f}",
        "max": f"max({img_prob:.4f}, {txt_prob:.4f}) = {max(img_prob, txt_prob):.4f}",
        "min": f"min({img_prob:.4f}, {txt_prob:.4f}) = {min(img_prob, txt_prob):.4f}",
    }

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

    logs.append(("STEP 2: Fusion Strategy", f"Strategy: {strategy.upper()}"))
    logs.append(("STEP 2: Formula", strategy_formulas.get(strategy, "N/A")))
    logs.append(("STEP 2: Combined Score", f"P(Fake)_fused = {final:.4f}"))

    logs.append(("STEP 3: Thresholds", f"Fake >= 0.55 | Real <= 0.45 | Uncertain = 0.45 to 0.55"))
    logs.append(("STEP 3: Evaluation", f"{final:.4f} vs threshold 0.55"))
    logs.append(("STEP 3: Image Weight", f"Contribution: {img_prob:.4f} ({img_prob/(img_prob+txt_prob)*100:.1f}%)"))
    logs.append(("STEP 3: Text Weight", f"Contribution: {txt_prob:.4f} ({txt_prob/(img_prob+txt_prob)*100:.1f}%)"))

    if final >= 0.55:
        label, confidence = "Fake", final
    elif final <= 0.45:
        label, confidence = "Real", 1 - final
    else:
        label, confidence = "Uncertain", max(final, 1 - final)

    logs.append(("STEP 4: Final Verdict", f"{label} with confidence {confidence:.1%}"))

    return {
        "final_prediction": label,
        "final_confidence": confidence,
        "final_fake_probability": final,
        "final_real_probability": 1 - final,
        "image_contribution": img_prob,
        "text_contribution": txt_prob,
        "logs": logs,
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

                with st.expander("  Image Computation Breakdown", expanded=False):
                    for step_title, step_detail in image_result.get("logs", []):
                        st.markdown(f"**{step_title}**")
                        st.code(step_detail, language=None)

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

                with st.expander("  Text Computation Breakdown", expanded=False):
                    for step_title, step_detail in text_result.get("logs", []):
                        st.markdown(f"**{step_title}**")
                        st.code(step_detail, language=None)

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

                with st.expander("  Fusion Computation Breakdown", expanded=False):
                    for step_title, step_detail in combined.get("logs", []):
                        st.markdown(f"**{step_title}**")
                        st.code(step_detail, language=None)

        # ── FUSION BREAKDOWN ──
        if image_result and text_result:
            st.markdown('<div class="section-label">FUSION BREAKDOWN</div>', unsafe_allow_html=True)

            fb1, fb2, fb3, fb4 = st.columns(4)
            with fb1:
                st.metric("Image Score", f"{image_result['fake_probability']:.1%}", help="Fake probability from image branch")
            with fb2:
                st.metric("Text Score", f"{text_result['fake_probability']:.1%}", help="Fake probability from text branch")
            with fb3:
                st.metric("Strategy", fusion_strategy.upper())
            with fb4:
                st.metric("Final Score", f"{combined['final_fake_probability']:.1%}", help="Combined fake probability")

            st.progress(min(combined["final_fake_probability"], 1.0))

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
            verdict = entry["verdict"]
            verdict_color = {"Fake": "#ef4444", "Real": "#22c55e", "Uncertain": "#facc15"}.get(verdict, "#9ca3af")
            verdict_emoji = {"Fake": " ", "Real": " ", "Uncertain": " "}.get(verdict, " ")

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                 border-radius:10px; padding:1rem 1.25rem; margin-bottom:0.75rem;
                 border-left:4px solid {verdict_color};">
                <div style="color:rgba(255,255,255,0.3); font-size:0.7rem; margin-bottom:0.25rem;">{entry['time']}</div>
                <div style="color:{verdict_color}; font-size:1rem; font-weight:700;">
                    {verdict_emoji} #{i+1} — {verdict}
                </div>
                <div style="color:rgba(255,255,255,0.4); font-size:0.8rem; margin-top:0.25rem;">
                    {entry['text_preview'][:60]}{'...' if len(entry.get('text_preview','')) > 60 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            hcols = st.columns(4)
            with hcols[0]:
                if entry["image_prob"] is not None:
                    st.metric("Image", f"{entry['image_prob']:.1%}", help="Fake probability from image branch")
                else:
                    st.metric("Image", "N/A")
            with hcols[1]:
                if entry["text_prob"] is not None:
                    st.metric("Text", f"{entry['text_prob']:.1%}", help="Fake probability from text branch")
                else:
                    st.metric("Text", "N/A")
            with hcols[2]:
                st.metric("Strategy", entry.get("strategy", "N/A"))
            with hcols[3]:
                st.metric("Final", f"{entry['final_prob']:.1%}", help="Combined fake probability")

            st.progress(min(entry["final_prob"], 1.0))
            st.markdown("")

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
