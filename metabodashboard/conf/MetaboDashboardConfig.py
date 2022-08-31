NUMBER_METABO_TO_KEEP = 15

from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix


def true_positive_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[1][1]


def false_positive_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[0][1]


def true_negative_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[0][0]


def false_negative_rate(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)[1][0]


STATISTICS = {
    "Accuracy": accuracy_score,
    # "f1-score": f1_score,
    # "Precision": precision_score,
    "True positive": true_positive_rate,
    "False positive": false_positive_rate,
    "True negative": true_negative_rate,
    "False negative": false_negative_rate,
}
