from argparse import ArgumentParser
from time import time_ns
import os
import json
from commode_utils.metrics import ClassificationMetrics
from scripts.code2seq.fine_tune import train_and_test as c2s_train_and_test
from scripts.treelstm.fine_tune import train_and_test as tl_train_and_test
from scripts.code2seq.dump_results import extract as c2s_extract
from scripts.treelstm.dump_results import extract as tl_extract
from scripts.save_metrics import calculate_and_dump_metrics
from scripts.utils import PREPROCESSED_DATASETS_DIR, EXPERIMENT_MODEL_DIR, RESULTS_DIR


def save_metrics(metrics: ClassificationMetrics, path: str) -> None:
    with open(path, "w") as output:
        d = {"f1": metrics.f1_score.item(), "precision": metrics.precision.item(), "recall": metrics.recall.item()}
        json.dump(d, output)


def save_metrics_and_predictions(run_name: str, project_name: str) -> None:
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    outdir = os.path.join(RESULTS_DIR, project_name)
    os.makedirs(outdir)

    full_name = os.path.join(EXPERIMENT_MODEL_DIR, run_name)
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


def run_models_and_save_results(project_name: str) -> None:
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)

    run_name = f"{project_name}_{time_ns()}"

    new_model_folder = os.path.join(EXPERIMENT_MODEL_DIR, run_name, "new")
    train_and_test(dataset_path, new_model_folder)

    trained_model_folder = os.path.join(EXPERIMENT_MODEL_DIR, run_name, "trained")
    train_and_test(dataset_path, trained_model_folder, MAIN_MODEL)

    save_metrics_and_predictions(run_name, project_name)


def evaluate_on_many_datasets(filename: str) -> None:
    """Evaluate models on each project's dataset"""

    with open(filename, "r") as projects_file:
        for project in projects_file:
            run_models_and_save_results(project.strip())


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("projects_file", type=str, help="A path to file with list of preprocessed projects' names")
    arg_parser.add_argument("model_type", type=str, help="Model type (code2seq, treelstm)")
    arg_parser.add_argument("model", type=str, help="Path to pretrained code2seq model")
    args = arg_parser.parse_args()

    MAIN_MODEL = args.model
    if args.model_type == "code2seq":
        train_and_test = c2s_train_and_test
        extract = c2s_extract
    elif args.model_type == "treelstm":
        train_and_test = tl_train_and_test
        extract = tl_extract
    else:
        raise ValueError("Unknown model")

    evaluate_on_many_datasets(args.projects_file)
