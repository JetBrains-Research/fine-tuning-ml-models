import json
import math

import numpy
import numpy as np
import ast
import matplotlib.pyplot as plt
from scripts.utils import RESULTS_DIR
import os
import pandas as pd


def save_metric_plot(metric_new, metric_trained_before, metric_trained_after, metric_name: str):
    metric_new = np.asarray(metric_new)
    metric_trained_before = np.asarray(metric_trained_before)
    metric_trained_after = np.asarray(metric_trained_after)

    fig, ax = plt.subplots()

    metric_name = metric_name.upper()
    ax.plot(numpy.arange(1, metric_new.shape[0] + 1), np.sort(metric_new), "o", label="From scratch")
    ax.plot(
        numpy.arange(1, metric_trained_before.shape[0] + 1), np.sort(metric_trained_before), "o", label="Pretrained"
    )
    ax.plot(numpy.arange(1, metric_trained_after.shape[0] + 1), np.sort(metric_trained_after), "o", label="Fine-tuned")
    ax.legend()

    plt.ylabel(metric_name)

    fig.suptitle(f"{metric_name} distribution on test parts", fontweight="bold")
    fig.savefig(f"{metric_name}_test.png")

    print(f"{metric_name} mean improved", np.mean(metric_trained_after - metric_trained_before))
    print(f"{metric_name} mean from scratch", np.mean(metric_new))


def save_f1_test_plot():
    f1_new = []
    f1_trained_before = []
    f1_trained_after = []
    for project in os.listdir(RESULTS_DIR):
        project_folder = os.path.join(RESULTS_DIR, project)

        with open(os.path.join(project_folder, "new_after.jsonl")) as f:
            metrics = ast.literal_eval(f.readline().replace('\'', "\"").replace("nan", "0"))
            f1_new.append(metrics["test/f1"])

        with open(os.path.join(project_folder, "trained_before.jsonl")) as f:
            metrics = ast.literal_eval(f.readline())
            f1_trained_before.append(metrics["test/f1"])

        with open(os.path.join(project_folder, "trained_after.jsonl")) as f:
            metrics = ast.literal_eval(f.readline())
            f1_trained_after.append(metrics["test/f1"])
            if metrics["test/f1"] > 0.99:
                print(project)

    save_metric_plot(f1_new, f1_trained_before, f1_trained_after, "f1")


def save_text_metric_plot(metric_name: str):
    metric_new = []
    metric_trained_before = []
    metric_trained_after = []
    for project in os.listdir(RESULTS_DIR):
        project_folder = os.path.join(RESULTS_DIR, project)

        df_new = pd.read_csv(os.path.join(project_folder, "new_after_metrics.csv"), index_col=0)
        val = df_new.at["means", metric_name]
        if math.isnan(val):
            val = 0
        metric_new.append(val)

        df_trained_before = pd.read_csv(os.path.join(project_folder, "trained_before_metrics.csv"), index_col=0)
        metric_trained_before.append(df_trained_before.at["means", metric_name])

        df_trained_after = pd.read_csv(os.path.join(project_folder, "trained_after_metrics.csv"), index_col=0)
        metric_trained_after.append(df_trained_after.at["means", metric_name])

    save_metric_plot(metric_new, metric_trained_before, metric_trained_after, metric_name)


if __name__ == "__main__":
    save_f1_test_plot()
    save_text_metric_plot("bleu")
    save_text_metric_plot("chrf")
    save_text_metric_plot("meteor")
