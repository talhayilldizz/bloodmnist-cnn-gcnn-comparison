import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> dict[str, float]:
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)

    return {
        "accuracy": float(accuracy_score(target_array, prediction_array)),
        "macro_f1": float(
            f1_score(
                target_array,
                prediction_array,
                average="macro",
                zero_division=0,
            )
        ),
    }