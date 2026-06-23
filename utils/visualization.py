import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
from typing import Dict, List, Optional


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plot a styled confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        annot_kws={"size": 16}
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()


def plot_roc_curve(
    y_true: List[int],
    y_prob: List[float],
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> float:
    """
    Plot ROC curve and return AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        color="#2196F3",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.2%})"
    )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()
    return float(roc_auc)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    title: str = "Training History",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r--", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, train_accs, "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_accs, "r--", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Curves", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a grouped bar chart comparing multiple models.
    """
    if metrics is None:
        metrics = ["accuracy", "fpr", "fnr", "f1"]
    
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    colors = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"]
    
    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in model_names]
        bars = ax.bar(x + i * width - 0.4 + width/2, values, width, 
                      label=metric.upper(), color=colors[i % len(colors)], alpha=0.85)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close()
