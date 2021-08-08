import os
import csv
from argparse import ArgumentParser
from .preprocess import preprocess_single
from .test_single import test_single
from .load_tools import setup_psiminer, setup_code2seq
from .utils import PREPROCESSED_DATASETS_DIR


def test_all(dataset_path: str, model_path: str, results_path: str):
    """Evaluate and store all"""

    project_names = os.listdir(dataset_path)
    for project_name in project_names:
        preprocess_single(os.path.join(dataset_path, project_name))

    project_names = os.listdir(PREPROCESSED_DATASETS_DIR)
    result_file = os.path.join(results_path, "results.csv")
    header = ["Project", "F1", "Precision", "Recall", "Loss"]
    with open(result_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for project_name in project_names:
            print(project_name)
            try:
                metrics = test_single(model_path, os.path.join(PREPROCESSED_DATASETS_DIR, project_name))
            except:
                metrics = [-1, -1, -1, -1]
            row = {"Project": project_name}
            for i in range(1, len(header)):
                row[header[i]] = metrics[i - 1]
            writer.writerow(row)
        f.close()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("dataset", type=str, help="Path to collection of Java projects")
    arg_parser.add_argument("model", type=str, help="Path to model")
    arg_parser.add_argument("results", type=str, help="Path to resulting .csv file")

    args = arg_parser.parse_args()

    setup_psiminer()
    setup_code2seq()
    test_all(args.dataset, args.model, args.results)
