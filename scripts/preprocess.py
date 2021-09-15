import os
import tempfile
from argparse import ArgumentParser
from shutil import copytree

from .load_tools import setup_psiminer
from .utils import PSIMINER_DIR, PREPROCESSED_DATASETS_DIR, PSIMINER_CODE2SEQ_CONFIG, PSIMINER_TREELSTM_CONFIG


def add_missing_files(dataset_path: str, extension: str) -> None:
    for file in [f"train.{extension}", f"val.{extension}", f"test.{extension}"]:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            open(file_path, "a").close()


def fix_c2s(dataset_path: str) -> None:
    add_missing_files(dataset_path, "c2s")

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


def fix_jsonl(dataset_path: str) -> None:
    add_missing_files(dataset_path, "jsonl")

    # TODO: duplicates filtering


def run_psiminer(source_folder: str, destination_folder: str, model_type: str) -> None:
    """Run psiminer and set correct filenames"""

    if model_type == "code2seq":
        cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{source_folder}" "{destination_folder}" {PSIMINER_CODE2SEQ_CONFIG}'
        os.system(cmd)
        fix_c2s(destination_folder)
    elif model_type == "treelstm":
        cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{source_folder}" "{destination_folder}" {PSIMINER_TREELSTM_CONFIG}'
        os.system(cmd)
        fix_jsonl(destination_folder)
    else:
        raise ValueError("Unknown model")


def preprocess_complete(project_path: str, model_type: str) -> None:
    """Transform project into test, train and val data for code2seq"""

    setup_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    run_psiminer(project_path, dataset_path, model_type)


def preprocess_single(project_path: str, model_type: str) -> None:
    """Transform project into test data for code2seq via psiminer"""

    setup_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        data_path = os.path.join(tmp, project_name)
        new_path = os.path.join(data_path, "test")
        os.makedirs(os.path.join(data_path, "train"))
        os.makedirs(os.path.join(data_path, "val"))
        copytree(project_path, new_path)
        preprocess_complete(data_path, model_type)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("projects", type=str, help="Path to file with projects to be preprocessed")
    arg_parser.add_argument("model_type", type=str, help="Model type (code2seq, treelstm)")
    args = arg_parser.parse_args()

    with open(args.projects, "r") as projects:
        for project in projects:
            preprocess_complete(project.strip(), args.model_type)
