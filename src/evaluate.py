from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)

    return report, roc_auc