from argparse import ArgumentParser
from time import time_ns
import os
from scripts.code2seq.fine_tune import train_and_test as c2s_train_and_test
#from scripts.treelstm.fine_tune import train_and_test as tl_train_and_test
from scripts.utils import PREPROCESSED_DATASETS_DIR, EXPERIMENT_MODEL_DIR


def run_models_and_save_results(project_name: str, model_type: str, model_path: str) -> None:
    if model_type == "code2seq":
        train_and_test = c2s_train_and_test
    # elif model_type == "treelstm":
    #     train_and_test = tl_train_and_test
    else:
        raise ValueError("Unknown model")

    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)

    run_name = f"{project_name}_{time_ns()}"

    new_model_folder = os.path.join(EXPERIMENT_MODEL_DIR, run_name, "new")
    train_and_test(dataset_path, new_model_folder)

    trained_model_folder = os.path.join(EXPERIMENT_MODEL_DIR, run_name, "trained")
    train_and_test(dataset_path, trained_model_folder, model_path)


def evaluate_on_many_datasets(filename: str, model_type: str, model_path: str) -> None:
    """Evaluate models on each project's dataset"""

    with open(filename, "r") as projects_file:
        for project in projects_file:
            run_models_and_save_results(project.strip(), model_type, model_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("projects_file", type=str, help="A path to file with list of preprocessed projects' names")
    arg_parser.add_argument("model_type", type=str, help="Model type (code2seq, treelstm)")
    arg_parser.add_argument("model", type=str, help="Path to pretrained code2seq model")
    args = arg_parser.parse_args()

    evaluate_on_many_datasets(args.projects_file, args.model_type, args.model)
