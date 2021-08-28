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


def fix_c2s(dataset_path: str) -> None:
    for file in ["train.c2s", "val.c2s", "test.c2s"]:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            open(file_path, "a").close()

    with open(os.path.join(dataset_path, "train.c2s"), "r") as train:
        train_samples_set = set("".join(sorted(sample.split())) for sample in train)

    val_samples = []
    val_samples_set = set()
    with open(os.path.join(dataset_path, "val.c2s"), "r") as val:
        for sample in val:
            paths = "".join(sorted(sample.split()))
            if paths not in train_samples_set:
                val_samples.append(sample)
                val_samples_set.add(paths)
    with open(os.path.join(dataset_path, "val.c2s"), "w") as val:
        val.writelines(val_samples)

    test_samples = []
    test_samples_set = set()
    with open(os.path.join(dataset_path, "test.c2s"), "r") as test:
        for sample in test:
            paths = "".join(sorted(sample.split()))
            if paths not in train_samples_set and paths not in val_samples_set:
                test_samples.append(sample)
                test_samples_set.add(paths)
    with open(os.path.join(dataset_path, "test.c2s"), "w") as test:
        test.writelines(test_samples)

    print("Train:", len(train_samples_set))
    print("Val:", len(val_samples_set))
    print("Test:", len(test_samples_set))


def preprocess_complete(project_path: str) -> None:
    """Transform project into test, train and val data for code2seq"""

    setup_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    run_psiminer(project_path, dataset_path)
    fix_c2s(dataset_path)


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
    arg_parser.add_argument("projects", type=str, help="Path to file with projects to be preprocessed")

    args = arg_parser.parse_args()

    with open(args.projects, "r") as projects:
        for project in projects:
            preprocess_complete(project.strip())
