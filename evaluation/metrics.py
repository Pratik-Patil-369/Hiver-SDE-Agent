from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_intent(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_precision": float(prec),
            "macro_recall": float(rec), "macro_f1": float(f1),
            "n": len(y_true)}


def per_class_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0)


def save_confusion_matrix(y_true, y_pred, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
