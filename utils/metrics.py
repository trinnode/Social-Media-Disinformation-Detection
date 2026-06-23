import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Returns:
        Dictionary containing accuracy, FPR, FNR, precision, recall, F1, and AUC.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tnr": float(tnr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["auc"] = float(auc)
        except ValueError:
            metrics["auc"] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Print formatted evaluation metrics.
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} Performance Metrics")
    print(f"{'='*50}")
    print(f"  Accuracy:           {metrics['accuracy']:.2%}")
    print(f"  Precision:          {metrics['precision']:.2%}")
    print(f"  Recall:             {metrics['recall']:.2%}")
    print(f"  F1-Score:           {metrics['f1']:.2%}")
    print(f"  False Positive Rate: {metrics['fpr']:.2%}")
    print(f"  False Negative Rate: {metrics['fnr']:.2%}")
    if "auc" in metrics:
        print(f"  AUC-ROC:            {metrics['auc']:.2%}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {metrics['tp']:3d}  |  FP: {metrics['fp']:3d}")
    print(f"    FN: {metrics['fn']:3d}  |  TN: {metrics['tn']:3d}")
    print(f"{'='*50}\n")


def compare_models(results: Dict[str, Dict[str, float]]) -> None:
    """
    Compare performance metrics across multiple models.
    """
    print(f"\n{'='*80}")
    print(f"  Model Comparison")
    print(f"{'='*80}")
    
    header = f"{'Model':<25} {'Accuracy':>10} {'FPR':>10} {'FNR':>10} {'F1':>10} {'AUC':>10}"
    print(header)
    print("-" * 80)
    
    for name, metrics in results.items():
        auc = metrics.get("auc", 0.0)
        row = (
            f"{name:<25} "
            f"{metrics['accuracy']:>10.2%} "
            f"{metrics['fpr']:>10.2%} "
            f"{metrics['fnr']:>10.2%} "
            f"{metrics['f1']:>10.2%} "
            f"{auc:>10.2%}"
        )
        print(row)
    
    print(f"{'='*80}\n")
