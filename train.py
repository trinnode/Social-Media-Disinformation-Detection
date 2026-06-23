"""
Training Pipeline for Social Media Disinformation Detection
Handles training of image branch, text branch, and fusion models.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import config
from models.image_branch import ImageBranch, ImageBranchWithELA
from models.text_branch import TextBranch
from models.fusion import (
    ScoreAveragingFusion,
    XGBoostFusion,
    NeuralNetworkFusion,
    DualBranchClassifier
)
from utils.dataset import DisinformationDataset, create_dataloaders
from utils.metrics import compute_metrics, print_metrics, compare_models
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve
from data.generate_dataset import load_dataset, split_dataset, generate_dataset


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Trainer:
    """
    Complete training pipeline for the dual-branch classifier.
    """
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.models_dir = Path("models/checkpoints")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            config.text.model_name
        )
        
        self.image_transform = transforms.Compose([
            transforms.Resize(config.image.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(config.image.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data loaders."""
        print("\nPreparing dataset...")
        
        posts = load_dataset()
        train_data, val_data, test_data = split_dataset(posts)
        
        print(f"  Train: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data,
            val_data,
            test_data,
            text_tokenizer=self.tokenizer,
            image_transform=self.image_transform,
            batch_size=config.train.image_batch_size,
            num_workers=config.train.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def train_image_branch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[ImageBranch, Dict]:
        """Train the image classification branch."""
        print("\n" + "="*60)
        print("  Training Image Branch (ResNet-18 + ELA)")
        print("="*60)
        
        model = ImageBranchWithELA(num_classes=2, pretrained=True).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.image_lr,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        
        early_stopping = EarlyStopping(patience=5)
        best_val_acc = 0.0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        for epoch in range(config.train.image_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.train.image_epochs}")
            for batch in pbar:
                original = batch["image"].to(self.device)
                ela = batch["ela_image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(original, ela)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    original = batch["image"].to(self.device)
                    ela = batch["ela_image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = model(original, ela)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    self.models_dir / "image_branch_best.pth"
                )
                print(f"  -> Saved best model (val_acc: {val_acc:.2%})")
            
            if early_stopping(val_loss):
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        
        model.load_state_dict(
            torch.load(self.models_dir / "image_branch_best.pth", weights_only=True)
        )
        
        return model, history
    
    def train_text_branch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[TextBranch, Dict]:
        """Train the text classification branch."""
        print("\n" + "="*60)
        print("  Training Text Branch (DistilBERT)")
        print("="*60)
        
        model = TextBranch(
            model_name=config.text.model_name,
            num_classes=2,
            dropout=0.3
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.text.learning_rate,
            weight_decay=config.text.weight_decay
        )
        
        early_stopping = EarlyStopping(patience=5)
        best_val_acc = 0.0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        for epoch in range(config.text.epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.text.epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    self.models_dir / "text_branch_best.pth"
                )
                print(f"  -> Saved best model (val_acc: {val_acc:.2%})")
            
            if early_stopping(val_loss):
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        
        model.load_state_dict(
            torch.load(self.models_dir / "text_branch_best.pth", weights_only=True)
        )
        
        return model, history
    
    def evaluate_branch(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        branch_type: str = "image"
    ) -> Dict:
        """Evaluate a single branch on test data."""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                labels = batch["label"].to(self.device)
                
                if branch_type == "image":
                    original = batch["image"].to(self.device)
                    ela = batch["ela_image"].to(self.device)
                    outputs = model(original, ela)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = model(input_ids, attention_mask)
                
                probs = torch.softmax(outputs, dim=-1)[:, 1]
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return metrics, all_preds, all_labels, all_probs
    
    def extract_branch_probabilities(
        self,
        image_model: ImageBranch,
        text_model: TextBranch,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract probability scores from both branches for fusion training."""
        image_model.eval()
        text_model.eval()
        
        image_probs = []
        text_probs = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                original = batch["image"].to(self.device)
                ela = batch["ela_image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"]
                
                img_out = image_model(original, ela)
                txt_out = text_model(input_ids, attention_mask)
                
                img_probs = torch.softmax(img_out, dim=-1)[:, 1]
                txt_probs = torch.softmax(txt_out, dim=-1)[:, 1]
                
                image_probs.extend(img_probs.cpu().numpy())
                text_probs.extend(txt_probs.cpu().numpy())
                labels.extend(batch_labels.numpy())
        
        return (
            np.array(image_probs),
            np.array(text_probs),
            np.array(labels)
        )
    
    def train_fusion_models(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        image_model: ImageBranch,
        text_model: TextBranch
    ) -> Dict:
        """Train and evaluate all fusion strategies."""
        print("\n" + "="*60)
        print("  Training Fusion Strategies")
        print("="*60)
        
        train_img_probs, train_txt_probs, train_labels = \
            self.extract_branch_probabilities(image_model, text_model, train_loader)
        
        val_img_probs, val_txt_probs, val_labels = \
            self.extract_branch_probabilities(image_model, text_model, val_loader)
        
        fusion_results = {}
        
        print("\n1. Score Averaging Fusion")
        avg_fusion = ScoreAveragingFusion()
        avg_preds = ((val_img_probs + val_txt_probs) / 2 > 0.5).astype(int)
        avg_metrics = compute_metrics(val_labels, avg_preds, (val_img_probs + val_txt_probs) / 2)
        fusion_results["Score Averaging"] = avg_metrics
        print_metrics(avg_metrics, "Score Averaging")
        
        print("\n2. XGBoost Stacking Fusion")
        xgb_fusion = XGBoostFusion()
        xgb_fusion.train(train_img_probs, train_txt_probs, train_labels)
        xgb_preds = xgb_fusion.predict(val_img_probs, val_txt_probs)
        xgb_probs = xgb_fusion.predict_proba(val_img_probs, val_txt_probs)
        xgb_metrics = compute_metrics(val_labels, xgb_preds, xgb_probs)
        fusion_results["XGBoost Stacking"] = xgb_metrics
        print_metrics(xgb_metrics, "XGBoost Stacking")
        
        print("\n3. Neural Network Fusion")
        nn_fusion = NeuralNetworkFusion(input_dim=2, hidden_dim=64).to(self.device)
        nn_optimizer = optim.Adam(nn_fusion.parameters(), lr=0.001)
        nn_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(50):
            nn_fusion.train()
            nn_optimizer.zero_grad()
            
            features = torch.tensor(
                np.column_stack([train_img_probs, train_txt_probs]),
                dtype=torch.float32
            ).to(self.device)
            labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(self.device)
            
            outputs = nn_fusion(features)
            loss = nn_criterion(outputs, labels_tensor)
            loss.backward()
            nn_optimizer.step()
        
        nn_fusion.eval()
        with torch.no_grad():
            val_features = torch.tensor(
                np.column_stack([val_img_probs, val_txt_probs]),
                dtype=torch.float32
            ).to(self.device)
            nn_outputs = nn_fusion(val_features)
            nn_probs = torch.softmax(nn_outputs, dim=-1)[:, 1].cpu().numpy()
            nn_preds = np.argmax(nn_outputs.cpu().numpy(), axis=1)
        
        nn_metrics = compute_metrics(val_labels, nn_preds, nn_probs)
        fusion_results["Neural Network"] = nn_metrics
        print_metrics(nn_metrics, "Neural Network Fusion")
        
        torch.save({
            "xgboost_model": xgb_fusion,
            "nn_fusion_state": nn_fusion.state_dict(),
            "avg_weights": avg_fusion.weights.data.cpu().numpy()
        }, self.models_dir / "fusion_models.pth")
        
        return fusion_results
    
    def train_all(self) -> Dict:
        """Complete training pipeline."""
        print("\n" + "="*60)
        print(f"  {config.project_name}")
        print(f"  Version {config.version}")
        print("="*60)
        
        train_loader, val_loader, test_loader = self.prepare_data()
        
        image_model, image_history = self.train_image_branch(train_loader, val_loader)
        
        text_model, text_history = self.train_text_branch(train_loader, val_loader)
        
        image_metrics, _, _, _ = self.evaluate_branch(image_model, test_loader, "image")
        print_metrics(image_metrics, "Image Branch (Test)")
        
        text_metrics, _, _, _ = self.evaluate_branch(text_model, test_loader, "text")
        print_metrics(text_metrics, "Text Branch (Test)")
        
        fusion_results = self.train_fusion_models(
            train_loader, val_loader, image_model, text_model
        )
        
        all_results = {
            "Image Branch": image_metrics,
            "Text Branch": text_metrics,
            **fusion_results
        }
        
        compare_models(all_results)
        
        plot_training_history(
            image_history["train_loss"],
            image_history["val_loss"],
            image_history["train_acc"],
            image_history["val_acc"],
            title="Image Branch Training",
            save_path=str(self.results_dir / "image_training.png")
        )
        
        plot_training_history(
            text_history["train_loss"],
            text_history["val_loss"],
            text_history["train_acc"],
            text_history["val_acc"],
            title="Text Branch Training",
            save_path=str(self.results_dir / "text_training.png")
        )
        
        with open(self.results_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {self.results_dir}")
        print(f"Models saved to {self.models_dir}")
        
        return all_results


def main():
    """Main entry point for training."""
    trainer = Trainer()
    results = trainer.train_all()
    return results


if __name__ == "__main__":
    main()
