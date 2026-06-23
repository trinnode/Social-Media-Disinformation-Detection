import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.preprocessing import clean_text
from utils.ela import compute_ela


class DisinformationDataset(Dataset):
    """
    Dataset for social media disinformation detection.
    Handles both image and text modalities.
    """
    
    def __init__(
        self,
        data: List[Dict],
        image_transform=None,
        text_tokenizer=None,
        max_length: int = 128,
        is_test: bool = False
    ):
        self.data = data
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        image_path = item.get("image_path", "")
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        ela_image = compute_ela(image)
        
        if self.image_transform:
            image = self.image_transform(image)
            ela_image = self.image_transform(ela_image)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
            ela_image = transform(ela_image)
        
        text = item.get("text", "")
        cleaned_text = clean_text(text)
        
        if self.text_tokenizer:
            encoding = self.text_tokenizer(
                cleaned_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        label = torch.tensor(item.get("label", 0), dtype=torch.long)
        
        return {
            "image": image,
            "ela_image": ela_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "text": text,
        }


def create_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    text_tokenizer,
    image_transform=None,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    """
    train_dataset = DisinformationDataset(
        train_data,
        image_transform=image_transform,
        text_tokenizer=text_tokenizer
    )
    
    val_dataset = DisinformationDataset(
        val_data,
        image_transform=image_transform,
        text_tokenizer=text_tokenizer
    )
    
    test_dataset = DisinformationDataset(
        test_data,
        image_transform=image_transform,
        text_tokenizer=text_tokenizer,
        is_test=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
