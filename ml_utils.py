import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def auroc_train_and_val(model, X_train, X_val, train_targets, val_targets):
    """
    Compute and plot ROC curves for training and validation datasets.

    Parameters
    ----------
    model : sklearn-like estimator
        Trained model with predict_proba method.
    X_train : array-like
        Training feature matrix.
    X_val : array-like
        Validation feature matrix.
    train_targets : array-like
        True labels for the training set.
    val_targets : array-like
        True labels for the validation set.

    Returns
    -------
    None
        Displays ROC curve plots and prints AUC values.
    """

    # Predict probabilities for the positive class
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)

    # Compute ROC curves
    fpr_train, tpr_train, _ = roc_curve(train_targets, train_probs[:, 1])
    fpr_val, tpr_val, _ = roc_curve(val_targets, val_probs[:, 1])

    # Compute AUC scores
    auc_train = roc_auc_score(train_targets, train_probs[:, 1])
    auc_val = roc_auc_score(val_targets, val_probs[:, 1])

    # Plot ROC curves
    plt.figure(figsize=(6, 4))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.4f})")
    plt.plot(fpr_val, tpr_val, label=f"Validation ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='green')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def predict_and_plot(model, inputs, targets, name=''):
    """
    Predict class labels and probabilities, compute evaluation metrics,
    and plot the normalized confusion matrix.

    Parameters
    ----------
    model : sklearn-like estimator
        Trained classifier with predict and predict_proba methods.
    inputs : array-like
        Input feature matrix for evaluation.
    targets : array-like
        True target labels.
    name : str, optional
        Dataset name used in plot titles.

    Returns
    -------
    None
        Displays confusion matrix plot and prints a classification report.
    """

    # Predicted probabilities and class labels
    pred_prob = model.predict_proba(inputs)
    predictions = model.predict(inputs)

    # Compute normalized confusion matrix
    cm = confusion_matrix(targets, predictions, normalize='true')

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

    # Generate classification report and convert to DataFrame
    report = classification_report(targets, predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Print report with fixed decimal formatting
    print(df_report.round(3))
