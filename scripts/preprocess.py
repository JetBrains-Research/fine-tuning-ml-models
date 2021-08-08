import os
import tempfile
from argparse import ArgumentParser
from shutil import copytree

from .load_tools import setup_psiminer
from .utils import PSIMINER_DIR, PSIMINER_CONFIG, NO_TYPES_PATH


def fix_naming(dataset_path: str) -> None:
    """Fix names for code2seq compatability"""

    for old_name in os.listdir(dataset_path):
        if old_name == "nodes_vocabulary.csv":
            continue
        old_path = os.path.join(dataset_path, old_name)
        new_path = os.path.join(dataset_path, f"{NO_TYPES_PATH}.{old_name}")
        os.rename(old_path, new_path)


def run_psiminer(source_folder: str, destination_folder: str) -> None:
    """Run psiminer and set correct filenames"""

    cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{source_folder}" "{destination_folder}" {PSIMINER_CONFIG}'
    os.system(cmd)
    fix_naming(destination_folder)


def preprocess_complete(project_path: str) -> None:
    """Transform project into test, train and val data for code2seq"""

    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join("datasets", project_name, NO_TYPES_PATH)
    run_psiminer(project_path, dataset_path)


def preprocess_single(project_path: str) -> None:
    """Transform project into test data for code2seq via psiminer"""

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

    setup_psiminer()
    preprocess_single(args.project)
