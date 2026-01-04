import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    model = DecisionTreeClassifier(
        max_depth=max_depth, # baseline
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_decision_tree(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
