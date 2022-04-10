import os
import tempfile
import json
from argparse import ArgumentParser
from typing import List
from shutil import copytree

from scripts.load_tools import setup_psiminer
from scripts.utils import PSIMINER_DIR, PREPROCESSED_DATASETS_DIR, PSIMINER_CODE2SEQ_CONFIG, PSIMINER_TREELSTM_CONFIG


def add_missing_files(dataset_path: str, extension: str) -> None:
    for file in [f"train.{extension}", f"val.{extension}", f"test.{extension}"]:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            open(file_path, "a").close()


def dfs(vertex, tree: List) -> str:
    subtree = "{"
    vertex_tokens = f"{vertex['token']},{vertex['nodeType']},{vertex['tokenType']}"
    subtree += vertex_tokens
    for child in vertex["children"]:
        subtree += dfs(tree[child], tree)
    subtree += "}"
    return subtree


def sample_to_string(sample: str, extension: str) -> str:
    if extension == "c2s":
        return "".join(sorted(sample.split()))
    elif extension == "jsonl":
        tree = json.loads(sample)["tree"]
        return dfs(tree[0], tree)
    else:
        raise ValueError("Unknown extension")


def fix_output(dataset_path: str, extension: str) -> None:
    add_missing_files(dataset_path, extension)

    with open(os.path.join(dataset_path, f"train.{extension}"), "r") as train:
        train_samples_set = set(sample_to_string(sample, extension) for sample in train)

    val_samples = []
    val_samples_set = set()
    with open(os.path.join(dataset_path, f"val.{extension}"), "r") as val:
        for sample in val:
            paths = sample_to_string(sample, extension)
            if paths not in train_samples_set:
                val_samples.append(sample)
                val_samples_set.add(paths)
    with open(os.path.join(dataset_path, f"val.{extension}"), "w") as val:
        val.writelines(val_samples)

    test_samples = []
    test_samples_set = set()
    with open(os.path.join(dataset_path, f"test.{extension}"), "r") as test:
        for sample in test:
            paths = sample_to_string(sample, extension)
            if paths not in train_samples_set and paths not in val_samples_set:
                test_samples.append(sample)
                test_samples_set.add(paths)
    with open(os.path.join(dataset_path, f"test.{extension}"), "w") as test:
        test.writelines(test_samples)

    print("Train:", len(train_samples_set))
    print("Val:", len(val_samples_set))
    print("Test:", len(test_samples_set))


def run_psiminer(source_folder: str, destination_folder: str, config_path: str) -> None:
    """Run psiminer and set correct filenames"""

    cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{source_folder}" "{destination_folder}" {config_path}'
    os.system(cmd)
    if config_path == PSIMINER_TREELSTM_CONFIG:
        fix_output(destination_folder, "jsonl")
    elif config_path == PSIMINER_CODE2SEQ_CONFIG:
        fix_output(destination_folder, "c2s")


def preprocess_complete(project_path: str, config_path: str) -> None:
    """Transform project into test, train and val data for code2seq"""

    setup_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    run_psiminer(project_path, dataset_path, config_path)


def preprocess_single(project_path: str, config_path: str) -> None:
    """Transform project into test data for code2seq via psiminer"""

    setup_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    with tempfile.TemporaryDirectory(dir="..") as tmp:
        data_path = os.path.join(tmp, project_name)
        new_path = os.path.join(data_path, "test")
        os.makedirs(os.path.join(data_path, "train"))
        os.makedirs(os.path.join(data_path, "val"))
        copytree(project_path, new_path)
        preprocess_complete(data_path, config_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("projects", type=str, help="Path to file with projects to be preprocessed")
    arg_parser.add_argument("model_type", type=str, help="Model type (code2seq, treelstm)")
    args = arg_parser.parse_args()

    with open(args.projects, "r") as projects:
        for project in projects:
            if args.model_type == "code2seq":
                config = PSIMINER_CODE2SEQ_CONFIG
            elif args.model_type == "treelstm":
                config = PSIMINER_TREELSTM_CONFIG
            else:
                ValueError("Unknown model")
            preprocess_complete(project.strip(), config)
