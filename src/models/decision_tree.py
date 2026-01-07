import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score)

# Train a Desicion Tree classifier
def train_decision_tree(X_train, y_train, max_depth = None, random_state = 42):
    model = DecisionTreeClassifier(
        max_depth = max_depth, # baseline
        random_state = random_state
    )
    model.fit(X_train, y_train)
    return model

# Evaluate a trained Decision Tree classifier on test data
def evaluate_decision_tree(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division = 0),
        "recall": recall_score(y_test, y_pred, zero_division = 0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
