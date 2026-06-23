import re
import string
from typing import List, Optional
from PIL import Image
import numpy as np

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
}

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean text data by removing URLs, mentions, hashtags, and special characters.
    """
    if not isinstance(text, str):
        return ""
    
    text = URL_PATTERN.sub('', text)
    text = MENTION_PATTERN.sub('', text)
    text = HASHTAG_PATTERN.sub(' ', text)
    text = SPECIAL_CHAR_PATTERN.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    if remove_stopwords:
        words = text.lower().split()
        words = [w for w in words if w not in STOP_WORDS]
        text = ' '.join(words)
    
    return text


def extract_text_features(text: str) -> dict:
    """
    Extract statistical features from text that may indicate AI generation.
    """
    if not isinstance(text, str) or len(text) == 0:
        return {
            "word_count": 0,
            "avg_word_length": 0,
            "sentence_count": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "uppercase_ratio": 0,
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    return {
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "sentence_count": len(sentences),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }


def preprocess_image(
    image: Image.Image,
    target_size: tuple = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input.
    Resizes, converts to numpy array, and optionally normalizes.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(image, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
    
    return img_array


def sanitize_input(text: str) -> str:
    """
    Sanitize text input to prevent injection attacks.
    """
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    return text.strip()
