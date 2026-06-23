import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ImageBranch(nn.Module):
    """
    Image classification branch using ResNet-18.
    Classifies images as real or artificially generated.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer."""
        x = self.backbone.avgpool(self.backbone.layer4(
            self.backbone.layer3(
                self.backbone.layer2(
                    self.backbone.layer1(
                        self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
                    )
                )
            )
        ))
        return torch.flatten(x, 1)


class ImageBranchWithELA(nn.Module):
    """
    Image branch that processes both original image and ELA representation.
    Uses a dual-input ResNet-18 architecture.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        self.original_branch = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        
        self.ela_branch = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        
        num_features = self.original_branch.fc.in_features
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.original_branch.fc = nn.Identity()
        self.ela_branch.fc = nn.Identity()
    
    def forward(self, original: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        orig_features = self.original_branch(original)
        ela_features = self.ela_branch(ela)
        
        combined = torch.cat([orig_features, ela_features], dim=1)
        
        return self.fusion_layer(combined)
    
    def get_features(self, original: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        """Extract combined features before classification."""
        orig_features = self.original_branch(original)
        ela_features = self.ela_branch(ela)
        return torch.cat([orig_features, ela_features], dim=1)
