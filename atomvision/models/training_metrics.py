"""Module for Classification Training Metrics."""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jarvis.db.jsonutils import dumpjson
import pprint


def log_confusion_matrix(val_evaluator, val_loader, n_classes=5):
    """Get confusion matrix."""
    classes = [str(c) for c in list(range(n_classes))]
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    try:
        cm = metrics["cm"]
    except Exception:
        raise Exception("Colormap not in metrics list.")
    cm = cm.numpy()
    np.savetxt("CM.txt", cm)
    cm = cm.astype(int)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plt.subplot()
    sns.heatmap(
        cm / cm.sum(axis=1)[:, np.newaxis],
        annot=True,
        ax=ax,
        fmt=".1%",
        square=True,
        cbar=False,
        cmap=sns.diverging_palette(20, 220, n=200),
    )
    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(classes, rotation=90)
    ax.yaxis.set_ticklabels(classes, rotation=0)
    plt.savefig("CM.png")
    plt.close()


def performance_traces(history, output_dir="."):
    """Get performance report."""
    plt.plot(history["train"]["accuracy"], label="Training Accuracy")
    plt.plot(history["validation"]["accuracy"], label="Validation Accuracy")
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(frameon=False)
    plt.savefig("Acc.png")
    plt.close()

    plt.plot(history["train"]["nll"], label="Training Loss")
    plt.plot(history["validation"]["nll"], label="Validation Loss")
    plt.xlabel("No. of Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.savefig("Loss.png")
    plt.close()
    print("history")
    print(pprint.pprint(history))
    dumpjson(
        filename=os.path.join(output_dir, "history_val.json"),
        data=history["validation"]["accuracy"],
    )
    dumpjson(
        filename=os.path.join(output_dir, "history_train.json"),
        data=history["train"]["accuracy"],
    )
