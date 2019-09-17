from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

def measure_performance(X, y, clf, show_accuracy=True, 
                    show_classification_report=True, show_confusion_matrix=True):
    """A function to measure the performance of the trained model."""
    y_pred = clf.predict(X)
    if show_accuracy:
        print("\nAccuracy: {0:.2f}%".format(100 * accuracy_score(y, y_pred)),"\n")

    if show_classification_report:
        print("Classification report")
        print(classification_report(y,y_pred),"\n")

    if show_confusion_matrix:
        print("Confusion matrix")
        print(confusion_matrix(y,y_pred),"\n")