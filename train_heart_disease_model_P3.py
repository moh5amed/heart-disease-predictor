import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the train and validation splits
X_train = pd.read_csv('X_train_P3.csv')
X_val = pd.read_csv('X_val_P3.csv')
y_train = pd.read_csv('y_train_P3.csv').values.ravel()
y_val = pd.read_csv('y_val_P3.csv').values.ravel()

# Hyperparameter grids
logreg_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [500, 1000]
}
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Grid search for Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg_grid = GridSearchCV(logreg, logreg_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
logreg_grid.fit(X_train, y_train)
logreg_best = logreg_grid.best_estimator_
y_pred_logreg = logreg_best.predict(X_val)
y_proba_logreg = logreg_best.predict_proba(X_val)[:, 1]

# Grid search for Random Forest
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_val)
y_proba_rf = rf_best.predict_proba(X_val)[:, 1]

# Evaluation function
def print_metrics(y_true, y_pred, y_proba, model_name, best_params):
    print(f"\n{model_name} Results:")
    print(f"Best Params: {best_params}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")

# Print metrics for both models
print_metrics(y_val, y_pred_logreg, y_proba_logreg, "Logistic Regression", logreg_grid.best_params_)
print_metrics(y_val, y_pred_rf, y_proba_rf, "Random Forest", rf_grid.best_params_)

# Save the best model based on ROC-AUC
roc_auc_logreg = roc_auc_score(y_val, y_proba_logreg)
roc_auc_rf = roc_auc_score(y_val, y_proba_rf)

if roc_auc_logreg >= roc_auc_rf:
    joblib.dump(logreg_best, 'best_heart_disease_model_P3.joblib')
    print("\nLogistic Regression saved as best_heart_disease_model_P3.joblib")
else:
    joblib.dump(rf_best, 'best_heart_disease_model_P3.joblib')
    print("\nRandom Forest saved as best_heart_disease_model_P3.joblib") 