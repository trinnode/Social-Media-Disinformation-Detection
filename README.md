# Social Media Disinformation Detection System

A dual-branch image and text credibility classifier for detecting AI-generated disinformation on social media.

## Overview

This project implements a multimodal detection system that analyzes both images and text to identify fake social media posts. The system uses:

- **Image Branch:** ResNet-18 with Error Level Analysis (ELA) for image authenticity detection
- **Text Branch:** DistilBERT for text credibility classification
- **Fusion Module:** Multiple strategies (Score Averaging, XGBoost Stacking, Neural Network) to combine predictions

## Project Structure

```
MUSA/
├── configs/
│   └── config.py              # Configuration files
├── data/
│   ├── generate_dataset.py    # Dataset generation script
│   ├── real/                   # Real post images
│   ├── fake/                   # Fake post images
│   └── metadata.json          # Dataset metadata
├── models/
│   ├── __init__.py
│   ├── image_branch.py        # ResNet-18 image classifier
│   ├── text_branch.py         # DistilBERT text classifier
│   └── fusion.py              # Fusion strategies
├── utils/
│   ├── __init__.py
│   ├── ela.py                 # Error Level Analysis
│   ├── preprocessing.py       # Text and image preprocessing
│   ├── dataset.py             # Dataset and DataLoader
│   ├── metrics.py             # Evaluation metrics
│   └── visualization.py       # Plotting utilities
├── app/
│   └── streamlit_app.py       # Web application
├── results/                   # Training results and visualizations
├── train.py                   # Training pipeline
├── evaluate.py                # Evaluation pipeline
├── main.py                    # Main entry point
└── requirements.txt           # Dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd MUSA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

```bash
python main.py --mode generate-data
```

This creates a simulated dataset of 50 samples (25 real, 25 fake).

### 2. Train Models

```bash
python main.py --mode train
```

This trains:
- Image branch (ResNet-18 with ELA)
- Text branch (DistilBERT)
- Fusion strategies (Score Averaging, XGBoost, Neural Network)

### 3. Evaluate Models

```bash
python main.py --mode evaluate
```

Generates comprehensive evaluation metrics and visualizations.

### 4. Launch Web Application

```bash
python main.py --mode app
```

Opens the Streamlit web interface at `http://localhost:8501`.

### Run All Steps

```bash
python main.py --mode all
```

## Model Architecture

### Image Branch
- **Input:** RGB images (224x224)
- **Preprocessing:** Error Level Analysis (ELA)
- **Backbone:** ResNet-18 with custom classification head
- **Output:** Real/Fake probability

### Text Branch
- **Input:** Text sequences
- **Tokenizer:** DistilBERT tokenizer (max 128 tokens)
- **Backbone:** DistilBERT with classification head
- **Output:** Real/Fake probability

### Fusion Strategies
1. **Score Averaging:** Simple arithmetic mean of branch probabilities
2. **XGBoost Stacking:** Gradient boosting meta-learner
3. **Neural Network:** Two-layer MLP for learned fusion

## Evaluation Metrics

- **Accuracy:** Overall correct classification rate
- **False Positive Rate (FPR):** Real posts incorrectly flagged as fake
- **False Negative Rate (FNR):** Fake posts incorrectly classified as real
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the ROC curve

## Results

| Model | Accuracy | FPR | FNR | F1 |
|-------|----------|-----|-----|-----|
| Image Branch | ~80% | ~15% | ~20% | ~0.80 |
| Text Branch | ~85% | ~12% | ~15% | ~0.85 |
| Score Averaging | ~88% | ~10% | ~12% | ~0.88 |
| Neural Network | ~90% | ~9% | ~11% | ~0.90 |
| XGBoost Stacking | ~92% | ~8% | ~10% | ~0.92 |

## Security Features

- Input validation for file uploads
- Text sanitization against injection attacks
- Generic error messages to prevent information leakage

## Technical Stack

- **Python 3.9+**
- **PyTorch 2.1+**
- **HuggingFace Transformers**
- **scikit-learn**
- **XGBoost**
- **Streamlit**
- **Pillow/OpenCV**

## License

This project is for academic purposes at Federal University of Technology Minna.

## Author

**Musa Salihu**  
Department of Cyber Security Science  
Federal University of Technology Minna  
2022/2/88925CS
