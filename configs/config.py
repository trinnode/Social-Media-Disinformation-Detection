"""
Configuration for Social Media Disinformation Detection System
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
APP_DIR = BASE_DIR / "app"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageConfig:
    input_size: Tuple[int, int] = (224, 224)
    ela_quality: int = 95
    channels: int = 3
    augmentation_prob: float = 0.5


@dataclass
class TextConfig:
    max_length: int = 128
    model_name: str = "distilbert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 20
    warmup_steps: int = 100
    weight_decay: float = 0.01


@dataclass
class FusionConfig:
    strategy: str = "xgboost"
    input_dim: int = 2
    hidden_dim: int = 64
    dropout: float = 0.3


@dataclass
class TrainConfig:
    image_lr: float = 1e-4
    image_batch_size: int = 8
    image_epochs: int = 15
    text_lr: float = 2e-5
    text_batch_size: int = 8
    text_epochs: int = 10
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8501
    max_upload_size_mb: int = 10
    allowed_image_types: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp"])


@dataclass
class Config:
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    app: AppConfig = field(default_factory=AppConfig)
    project_name: str = "Social Media Disinformation Detection"
    version: str = "1.0.0"


config = Config()
