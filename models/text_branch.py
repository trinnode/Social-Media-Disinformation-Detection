import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from typing import Optional, Tuple


class TextBranch(nn.Module):
    """
    Text classification branch using DistilBERT.
    Classifies text as human-written or machine-generated.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_base: bool = False
    ):
        super().__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        hidden_size = self.distilbert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through DistilBERT and classifier.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        return self.classifier(cls_output)
    
    def get_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract features before the classification head."""
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]
    
    def get_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get probability scores for each class."""
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)


class TextBranchWithFeatures(nn.Module):
    """
    Extended text branch that combines DistilBERT embeddings
    with handcrafted linguistic features.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        num_handcrafted_features: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.distilbert.config.hidden_size
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size + num_handcrafted_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        combined = torch.cat([cls_output, handcrafted_features], dim=1)
        
        return self.feature_fusion(combined)
