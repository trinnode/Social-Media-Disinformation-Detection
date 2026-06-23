"""
Evaluation Pipeline for Social Media Disinformation Detection
Comprehensive evaluation of trained models on test data.
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torchvision import transforms
from transformers import DistilBertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import config
from models.image_branch import ImageBranch, ImageBranchWithELA
from models.text_branch import TextBranch
from models.fusion import DualBranchClassifier, ScoreAveragingFusion
from utils.dataset import DisinformationDataset, create_dataloaders
from utils.metrics import compute_metrics, print_metrics, compare_models
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison
)
from data.generate_dataset import load_dataset, split_dataset


class Evaluator:
    """
    Comprehensive evaluation of the dual-branch classifier.
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
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            config.text.model_name
        )
        
        self.test_transform = transforms.Compose([
            transforms.Resize(config.image.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_models(self) -> Tuple[ImageBranch, TextBranch]:
        """Load trained models from checkpoints."""
        print("\nLoading trained models...")
        
        image_model = ImageBranchWithELA(num_classes=2, pretrained=False).to(self.device)
        image_checkpoint = self.models_dir / "image_branch_best.pth"
        if image_checkpoint.exists():
            image_model.load_state_dict(torch.load(image_checkpoint, map_location=self.device, weights_only=True))
            print("  Image branch loaded successfully")
        else:
            print("  WARNING: Image branch checkpoint not found")
        
        text_model = TextBranch(
            model_name=config.text.model_name,
            num_classes=2
        ).to(self.device)
        text_checkpoint = self.models_dir / "text_branch_best.pth"
        if text_checkpoint.exists():
            text_model.load_state_dict(torch.load(text_checkpoint, map_location=self.device, weights_only=True))
            print("  Text branch loaded successfully")
        else:
            print("  WARNING: Text branch checkpoint not found")
        
        return image_model, text_model
    
    def prepare_test_data(self):
        """Prepare test data loader."""
        posts = load_dataset()
        _, _, test_data = split_dataset(posts)
        
        print(f"\nTest set: {len(test_data)} samples")
        
        test_dataset = DisinformationDataset(
            test_data,
            image_transform=self.test_transform,
            text_tokenizer=self.tokenizer,
            is_test=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.train.image_batch_size,
            shuffle=False,
            num_workers=config.train.num_workers
        )
        
        return test_loader, test_data
    
    def evaluate_single_branch(
        self,
        model: torch.nn.Module,
        test_loader,
        branch_type: str
    ) -> Dict:
        """Evaluate a single branch."""
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
        
        return compute_metrics(all_labels, all_preds, all_probs), all_preds, all_labels, all_probs
    
    def evaluate_fusion_strategies(
        self,
        image_model: ImageBranch,
        text_model: TextBranch,
        test_loader
    ) -> Dict:
        """Evaluate all fusion strategies."""
        image_probs = []
        text_probs = []
        labels = []
        
        image_model.eval()
        text_model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
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
        
        image_probs = np.array(image_probs)
        text_probs = np.array(text_probs)
        labels = np.array(labels)
        
        results = {}
        
        avg_probs = (image_probs + text_probs) / 2
        avg_preds = (avg_probs > 0.5).astype(int)
        results["Score Averaging"] = compute_metrics(labels, avg_preds, avg_probs)
        
        fusion_checkpoint = self.models_dir / "fusion_models.pth"
        if fusion_checkpoint.exists():
            fusion_data = torch.load(fusion_checkpoint, map_location=self.device, weights_only=False)
            
            nn_fusion_state = fusion_data.get("nn_fusion_state")
            if nn_fusion_state:
                from models.fusion import NeuralNetworkFusion
                nn_fusion = NeuralNetworkFusion(input_dim=2).to(self.device)
                nn_fusion.load_state_dict(nn_fusion_state)
                nn_fusion.eval()
                
                with torch.no_grad():
                    features = torch.tensor(
                        np.column_stack([image_probs, text_probs]),
                        dtype=torch.float32
                    ).to(self.device)
                    nn_outputs = nn_fusion(features)
                    nn_probs = torch.softmax(nn_outputs, dim=-1)[:, 1].cpu().numpy()
                    nn_preds = np.argmax(nn_outputs.cpu().numpy(), axis=1)
                
                results["Neural Network"] = compute_metrics(labels, nn_preds, nn_probs)
        
        return results
    
    def run_full_evaluation(self) -> Dict:
        """Run comprehensive evaluation."""
        print("\n" + "="*60)
        print("  Full Evaluation Pipeline")
        print("="*60)
        
        image_model, text_model = self.load_models()
        test_loader, test_data = self.prepare_test_data()
        
        print("\nEvaluating Image Branch...")
        img_metrics, img_preds, img_labels, img_probs = \
            self.evaluate_single_branch(image_model, test_loader, "image")
        print_metrics(img_metrics, "Image Branch (ResNet-18 + ELA)")
        
        print("Evaluating Text Branch...")
        txt_metrics, txt_preds, txt_labels, txt_probs = \
            self.evaluate_single_branch(text_model, test_loader, "text")
        print_metrics(txt_metrics, "Text Branch (DistilBERT)")
        
        print("Evaluating Fusion Strategies...")
        fusion_results = self.evaluate_fusion_strategies(
            image_model, text_model, test_loader
        )
        
        for name, metrics in fusion_results.items():
            print_metrics(metrics, f"Fusion: {name}")
        
        all_results = {
            "Image Branch": img_metrics,
            "Text Branch": txt_metrics,
            **fusion_results
        }
        
        compare_models(all_results)
        
        print("\nGenerating visualizations...")
        
        plot_confusion_matrix(
            img_labels, img_preds,
            title="Image Branch Confusion Matrix",
            save_path=str(self.results_dir / "image_confusion_matrix.png")
        )
        
        plot_confusion_matrix(
            txt_labels, txt_preds,
            title="Text Branch Confusion Matrix",
            save_path=str(self.results_dir / "text_confusion_matrix.png")
        )
        
        plot_roc_curve(
            img_labels, img_probs,
            title="Image Branch ROC Curve",
            save_path=str(self.results_dir / "image_roc_curve.png")
        )
        
        plot_roc_curve(
            txt_labels, txt_probs,
            title="Text Branch ROC Curve",
            save_path=str(self.results_dir / "text_roc_curve.png")
        )
        
        plot_model_comparison(
            all_results,
            metrics=["accuracy", "f1", "fpr", "fnr"],
            save_path=str(self.results_dir / "model_comparison.png")
        )
        
        with open(self.results_dir / "evaluation_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved to {self.results_dir}")
        
        return all_results


def main():
    evaluator = Evaluator()
    results = evaluator.run_full_evaluation()
    return results


if __name__ == "__main__":
    main()
