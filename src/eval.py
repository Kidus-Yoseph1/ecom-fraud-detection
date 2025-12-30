from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_curve, auc)

def evaluate(model,X_test,y_test):
    rf_preds = model.predict(X_test)
    rf_probs = model.predict_proba(X_test)[:, 1]

    precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_probs)
    auc_pr_rf = auc(recall_rf, precision_rf)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Random Forest AUC-PR: {auc_pr_rf:.4f}")
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_preds))
