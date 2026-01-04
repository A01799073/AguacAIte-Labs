import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def train_svm(X_train, y_train, C=1.0, kernel="rbf"):
    model = SVC(
        C=C,
        kernel=kernel,
        gamma="scale" # Avoid numbers errors
    )
    model.fit(X_train, y_train)
    return model


def evaluate_svm(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
