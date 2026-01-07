import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def train_knn(X_train, y_train, k=5, metric = "euclidean"):
    """
    Train a KNN classifier.
    """
    model = KNeighborsClassifier(
        n_neighbors = k,
        metric = metric
    )
    model.fit(X_train, y_train)
    return model


def evaluate_knn(model, X_test, y_test):
    """
    Evaluate a trained KNN model.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "confusion_matrix": cm
    }
