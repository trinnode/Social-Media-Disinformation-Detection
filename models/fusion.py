import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class ScoreAveragingFusion(nn.Module):
    """
    Simple score averaging fusion strategy.
    Combines predictions from image and text branches by averaging their probabilities.
    """
    
    def __init__(self, weights: Tuple[float, float] = (0.5, 0.5)):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights))
    
    def forward(self, image_probs: torch.Tensor, text_probs: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.weights, dim=0)
        combined = weights[0] * image_probs + weights[1] * text_probs
        return combined
    
    def predict(self, image_probs: torch.Tensor, text_probs: torch.Tensor) -> torch.Tensor:
        combined = self.forward(image_probs, text_probs)
        return (combined > 0.5).long()


class XGBoostFusion:
    """
    XGBoost stacking fusion strategy.
    Uses XGBoost as a meta-learner to combine branch predictions.
    """
    
    def __init__(self, **xgb_params):
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "use_label_encoder": False
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self.is_trained = False
    
    def prepare_features(
        self,
        image_probs: np.ndarray,
        text_probs: np.ndarray
    ) -> np.ndarray:
        """Create feature matrix from branch predictions."""
        features = np.column_stack([
            image_probs,
            text_probs,
            np.abs(image_probs - text_probs),
            image_probs * text_probs,
            np.maximum(image_probs, text_probs),
            np.minimum(image_probs, text_probs)
        ])
        return features
    
    def train(
        self,
        image_probs: np.ndarray,
        text_probs: np.ndarray,
        labels: np.ndarray
    ):
        """Train the XGBoost meta-learner."""
        X = self.prepare_features(image_probs, text_probs)
        self.model.fit(X, labels)
        self.is_trained = True
    
    def predict(self, image_probs: np.ndarray, text_probs: np.ndarray) -> np.ndarray:
        """Predict using the trained XGBoost model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X = self.prepare_features(image_probs, text_probs)
        return self.model.predict(X)
    
    def predict_proba(self, image_probs: np.ndarray, text_probs: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X = self.prepare_features(image_probs, text_probs)
        return self.model.predict_proba(X)[:, 1]


class NeuralNetworkFusion(nn.Module):
    """
    Neural network fusion strategy.
    Combines features from both branches through a small neural network.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fusion_network(features)
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features)
        return torch.argmax(logits, dim=-1)


class DualBranchClassifier(nn.Module):
    """
    Complete dual-branch classifier combining image and text branches
    with configurable fusion strategy.
    """
    
    def __init__(
        self,
        image_branch: nn.Module,
        text_branch: nn.Module,
        fusion_strategy: str = "neural_network",
        fusion_kwargs: dict = None
    ):
        super().__init__()
        
        self.image_branch = image_branch
        self.text_branch = text_branch
        self.fusion_strategy_name = fusion_strategy
        
        if fusion_strategy == "neural_network":
            self.fusion = NeuralNetworkFusion(**(fusion_kwargs or {"input_dim": 2}))
        elif fusion_strategy == "averaging":
            self.fusion = ScoreAveragingFusion()
        elif fusion_strategy == "xgboost":
            self.fusion = XGBoostFusion(**(fusion_kwargs or {}))
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        image_logits = self.image_branch(image)
        text_logits = self.text_branch(input_ids, attention_mask)
        
        image_probs = torch.softmax(image_logits, dim=-1)[:, 1]
        text_probs = torch.softmax(text_logits, dim=-1)[:, 1]
        
        if self.fusion_strategy_name == "xgboost":
            return torch.stack([image_probs, text_probs], dim=1)
        
        features = torch.stack([image_probs, text_probs], dim=1)
        return self.fusion(features)
    
    def predict(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed predictions including individual branch outputs.
        """
        image_logits = self.image_branch(image)
        text_logits = self.text_branch(input_ids, attention_mask)
        
        image_probs = torch.softmax(image_logits, dim=-1)
        text_probs = torch.softmax(text_logits, dim=-1)
        
        image_pred = torch.argmax(image_probs, dim=-1)
        text_pred = torch.argmax(text_probs, dim=-1)
        
        if self.fusion_strategy_name == "xgboost":
            combined_probs = torch.stack([image_probs[:, 1], text_probs[:, 1]], dim=1)
        else:
            features = torch.stack([image_probs[:, 1], text_probs[:, 1]], dim=1)
            combined_logits = self.fusion(features)
            combined_probs = torch.softmax(combined_logits, dim=-1)
        
        if isinstance(combined_probs, torch.Tensor) and combined_probs.dim() == 2:
            final_pred = torch.argmax(combined_probs, dim=-1)
            final_prob = combined_probs[:, 1]
        else:
            final_pred = (combined_probs > 0.5).long()
            final_prob = combined_probs
        
        return {
            "final_prediction": final_pred,
            "final_probability": final_prob,
            "image_prediction": image_pred,
            "image_probabilities": image_probs,
            "text_prediction": text_pred,
            "text_probabilities": text_probs
        }
