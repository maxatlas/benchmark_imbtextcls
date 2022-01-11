from sklearn.metrics import (f1_score,
                             roc_curve,
                             auc,
                             recall_score,
                             precision_score,
                             classification_report)


def metrics_frame(preds, labels, label_names):
    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_res = auc(fpr, tpr)
    cr = classification_report(labels, preds,)
                               # labels=list(range(len(label_names))), target_names=label_names)

    model_metrics = {
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