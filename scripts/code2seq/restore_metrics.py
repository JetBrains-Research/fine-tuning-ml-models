import os
import json
from .dump_results import extract
from commode_utils.metrics import ClassificationMetrics
from ..save_metrics import calculate_and_dump_metrics
from argparse import ArgumentParser


def save_metrics(metrics: ClassificationMetrics, path: str) -> None:
    with open(path, "w") as output:
        d = {"f1": metrics.f1_score.item(), "precision": metrics.precision.item(), "recall": metrics.recall.item()}
        json.dump(d, output)


arg_parser = ArgumentParser()
arg_parser.add_argument("projects_file", type=str, help="A path to file with list of preprocessed projects' names")
args = arg_parser.parse_args()

MAIN_MODEL = "models/code2seq.ckpt"

with open(args.projects_file, "r") as f:
    for line in f:
        project = line.strip()
        name = project.split("_")[0]
        dataset_path = os.path.join("datasets", name)
        outdir = os.path.join("results", name)
        os.makedirs(outdir)

        full_name = os.path.join("models", "fine_tuning_experiments", project)
        vocabulary = os.path.join(dataset_path, "vocabulary.pkl")
        new_model_name = os.listdir(os.path.join(full_name, "new"))[0]
        new_model_path = os.path.join(full_name, "new", new_model_name)

        new = extract(
            new_model_path, dataset_path, True, vocabulary, result_file=os.path.join(outdir, "new_after_names.txt")
        )
        calculate_and_dump_metrics(
            os.path.join(outdir, "new_after_names.txt"), os.path.join(outdir, "new_after_metrics.csv")
        )
        save_metrics(new, os.path.join(outdir, "new_after.jsonl"))

        before = extract(MAIN_MODEL, dataset_path, False, result_file=os.path.join(outdir, "trained_before_names.txt"))
        calculate_and_dump_metrics(
            os.path.join(outdir, "trained_before_names.txt"), os.path.join(outdir, "trained_before_metrics.csv")
        )
        save_metrics(before, os.path.join(outdir, "trained_before.jsonl"))

        trained_model_name = os.listdir(os.path.join(full_name, "trained"))[0]
        trained_model_path = os.path.join(full_name, "trained", trained_model_name)

        after = extract(
            trained_model_path, dataset_path, False, result_file=os.path.join(outdir, "trained_after_names.txt")
        )
        calculate_and_dump_metrics(
            os.path.join(outdir, "trained_after_names.txt"), os.path.join(outdir, "trained_after_metrics.csv")
        )
        save_metrics(after, os.path.join(outdir, "trained_after.jsonl"))
