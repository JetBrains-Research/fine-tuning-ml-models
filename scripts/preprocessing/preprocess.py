import os
import tempfile
import json
from argparse import ArgumentParser
from typing import List
from shutil import copytree

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


def run_psiminer(source_folder: str, destination_folder: str, model_type: str) -> None:
    """Run psiminer and set correct filenames"""

    if model_type == "code2seq":
        cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{source_folder}" "{destination_folder}" {PSIMINER_CODE2SEQ_CONFIG}'
        os.system(cmd)
        fix_output(destination_folder, "c2s")
    elif model_type == "treelstm":
        cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{source_folder}" "{destination_folder}" {PSIMINER_TREELSTM_CONFIG}'
        os.system(cmd)
        fix_output(destination_folder, "jsonl")
    else:
        raise ValueError("Unknown model")


def preprocess_complete(project_path: str, model_type: str) -> None:
    """Transform project into test, train and val data for code2seq"""

    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    run_psiminer(project_path, dataset_path, model_type)


def preprocess_single(project_path: str, model_type: str) -> None:
    """Transform project into test data for code2seq via psiminer"""

    project_name = os.path.basename(os.path.normpath(project_path))
    with tempfile.TemporaryDirectory(dir="..") as tmp:
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
