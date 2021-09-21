from argparse import ArgumentParser
import os
import tempfile
from shutil import copy

from code2seq.fine_tune import train_and_test
from .utils import NO_TYPES_PATH


def fine_tune_self(dataset_path: str, model_path: str, folds_number: int):
    """Do k-fold cross-validation and compare quality metrics before and after fine-tuning"""

    project_name = os.path.basename(os.path.normpath(dataset_path))
    start_path = os.path.join(dataset_path, NO_TYPES_PATH)
    dataset = open(os.path.join(start_path, f"{NO_TYPES_PATH}.test.c2s"), "r")
    samples = dataset.readlines()
    folds_number += 1
    fold_size = len(samples) // folds_number

    with tempfile.TemporaryDirectory(dir=".") as tmp, open("results.txt", "w") as result_file:
        for i in range(folds_number - 1):
            preprocessed_path = os.path.join(tmp, str(i + 1))
            fold_path = os.path.join(preprocessed_path, NO_TYPES_PATH)
            os.makedirs(fold_path)
            copy(os.path.join(start_path, "nodes_vocabulary.csv"), fold_path)

            with open(os.path.join(fold_path, f"{NO_TYPES_PATH}.train.c2s"), "w+") as train:
                train.writelines(samples[: i * fold_size])
                train.writelines(samples[(i + 2) * fold_size :])

            with open(os.path.join(fold_path, f"{NO_TYPES_PATH}.val.c2s"), "w+") as val:
                val.writelines(samples[(i + 1) * fold_size : (i + 2) * fold_size])

            with open(os.path.join(fold_path, f"{NO_TYPES_PATH}.test.c2s"), "w+") as test:
                test.writelines(samples[i * fold_size : (i + 1) * fold_size])

            print(f"Fold #{i}:", file=result_file)

            tuned_model_folder = os.path.join("models", "fine_tuned", project_name, str(i))
            trained_model_path, metrics_before, metrics_after = train_and_test(
                preprocessed_path,
                tuned_model_folder,
                model_path,
            )
            print("Metrics before:", metrics_before, file=result_file)
            print("Metrics after:", metrics_after, file=result_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")
    arg_parser.add_argument("folds", type=int, help="Number of folds for k-fold cross-validation")

    args = arg_parser.parse_args()
    fine_tune_self(args.project, args.model, args.folds)
