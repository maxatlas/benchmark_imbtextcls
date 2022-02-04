import torch
from sklearn.metrics import (
    f1_score,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    classification_report,
    accuracy_score)


def metrics_frame(probs, preds, labels, label_names):
    accuracy = accuracy_score(labels, preds)
    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")

    cr = classification_report(labels, preds, target_names=label_names)
    fpr, tpr, thresholds, auc_res = [-1] * 4

    if len(label_names) == 2:
        labels = [label.tolist().index(1) for label in labels]
        probs = torch.sigmoid(torch.tensor(probs))[:, 1]

        fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
        fpr, tpr, thresholds = fpr.tolist(), tpr.tolist(), thresholds.tolist()

        auc_res = auc(fpr, tpr)

    model_metrics = {
        "Accuracy": accuracy,
        "Precision, Micro": precision_micro,
        "Precision, Macro": precision_macro,
        "Recall, Micro": recall_micro,
        "Recall, Macro": recall_macro,
        "F1 score, Micro": f1_micro,
        "F1 score, Macro": f1_macro,
        "ROC curve": (fpr, tpr, thresholds),
        "AUC": auc_res,
        "Classification report": cr,
    }

    return model_metrics
