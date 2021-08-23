from argparse import ArgumentParser
from time import time_ns
import os
from scripts.fine_tune import train_and_test
from scripts.dump_results import extract
from scripts.save_metrics import calculate_and_dump_metrics
from scripts.utils import PREPROCESSED_DATASETS_DIR, RESULTS_DIR, EXPERIMENT_MODEL_DIR


def run_models_and_save_results(project_name: str, model_path: str) -> None:
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)

    run_name = f"{project_name}_{time_ns()}"
    result_folder = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(result_folder)

    new_model_folder = os.path.join(EXPERIMENT_MODEL_DIR, run_name, "new")
    os.makedirs(new_model_folder)
    new_path, new_before, new_after = train_and_test(dataset_path, new_model_folder)
    with open(os.path.join(result_folder, "new_after.jsonl"), "w") as f:
        print(*new_after, file=f)
    new_after_names = extract(new_path, dataset_path, 1, os.path.join(result_folder, "new_after_names.txt"))
    calculate_and_dump_metrics(new_after_names, os.path.join(result_folder, "new_after_metrics.csv"))

    trained_model_folder = os.path.join(EXPERIMENT_MODEL_DIR, run_name, "trained")
    os.makedirs(trained_model_folder)
    trained_path, trained_before, trained_after = train_and_test(dataset_path, trained_model_folder, model_path)
    with open(os.path.join(result_folder, "trained_before.jsonl"), "w") as f:
        print(*trained_before, file=f)
    trained_before_names = extract(model_path, dataset_path, 1, os.path.join(result_folder, "trained_before_names.txt"))
    calculate_and_dump_metrics(trained_before_names, os.path.join(result_folder, "trained_before_metrics.csv"))
    with open(os.path.join(result_folder, "trained_after.jsonl"), "w") as f:
        print(*trained_after, file=f)
    trained_after_names = extract(trained_path, dataset_path, 1, os.path.join(result_folder, "trained_after_names.txt"))
    calculate_and_dump_metrics(trained_after_names, os.path.join(result_folder, "trained_after_metrics.csv"))


def evaluate_on_many_datasets(filename: str, model_path: str) -> None:
    """Evaluate models on each project's dataset"""

    with open(filename, "r") as projects_file:
        for project in projects_file:
            run_models_and_save_results(project.strip(), model_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("projects_file", type=str, help="A path to file with list of preprocessed projects' names")
    arg_parser.add_argument("model", type=str, help="Path to pretrained code2seq model")
    args = arg_parser.parse_args()

    evaluate_on_many_datasets(args.projects_file, args.model)
