import os
import tempfile
from argparse import ArgumentParser
from shutil import copytree

from .load_tools import setup_psiminer
from .utils import PSIMINER_DIR, PREPROCESSED_DATASETS_DIR, PSIMINER_CONFIG


def run_psiminer(source_folder: str, destination_folder: str) -> None:
    """Run psiminer and set correct filenames"""

    cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{source_folder}" "{destination_folder}" {PSIMINER_CONFIG}'
    os.system(cmd)


def preprocess_complete(project_path: str) -> None:
    """Transform project into test, train and val data for code2seq"""

    setup_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    run_psiminer(project_path, dataset_path)

    for file in ["train.c2s", "val.c2s", "test.c2s"]:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            open(file_path, "a").close()


def preprocess_single(project_path: str) -> None:
    """Transform project into test data for code2seq via psiminer"""

    setup_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        data_path = os.path.join(tmp, project_name)
        new_path = os.path.join(data_path, "test")
        os.makedirs(os.path.join(data_path, "train"))
        os.makedirs(os.path.join(data_path, "val"))
        copytree(project_path, new_path)
        preprocess_complete(data_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to project's files to be preprocessed")

    args = arg_parser.parse_args()

    preprocess_single(args.project)
