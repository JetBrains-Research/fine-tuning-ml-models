from argparse import ArgumentParser
import os
import json
from typing import List, Dict
from shutil import copytree
from scripts.utils import CLONED_REPOS_DIR, EXTRACTED_METHODS_DIR
from scripts.load_tools import setup_comment_updater
import git


def write_classes(methods_list: List[Dict[str, str]], folder: str) -> None:
    """Wrap methods into separate java classes and write them"""

    for i in range(len(methods_list)):
        filename = f"A{i}.java"
        with open(os.path.join(folder, filename), "w") as f:
            f.write(f'public class A{i} {"{"}\n')
            f.write(methods_list[i]["code"])
            f.write("}\n")


def filter_duplicates(methods: List) -> List:
    raw_codes = set()
    filtered = []
    for method in methods:
        clear_code = "".join(method["code"].split())
        if clear_code not in raw_codes:
            filtered.append(method)
            raw_codes.add(clear_code)
    return filtered


def split_dataset(project_name: str, train_part: float) -> str:
    """Process data mined by CommentUpdater in order to separate in train, test and validation samples"""

    dataset_dir = os.path.join(EXTRACTED_METHODS_DIR, project_name)
    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "val", project_name)
    os.makedirs(val_path)
    test_path = os.path.join(dataset_dir, "test", project_name)
    os.makedirs(test_path)

    raw_samples = open(os.path.join(dataset_dir, f"{project_name}.jsonl"), "r")
    added_methods = [sample for sample in list(map(json.loads, raw_samples)) if sample["update"] == "ADD"]
    added_methods.sort(key=lambda method: method["commitTime"])
    added_methods = filter_duplicates(added_methods)
    num_of_methods = len(added_methods)

    start_idx = int(train_part * num_of_methods) - 1
    snapshot_commit = added_methods[start_idx]["commitId"]
    source_dir = os.path.join(CLONED_REPOS_DIR, project_name)
    repo = git.Repo(source_dir)
    repo.head.reset(snapshot_commit, index=True, working_tree=True)
    copytree(source_dir, train_path)

    new_idx = start_idx + 1
    for idx in range(start_idx, num_of_methods):
        if added_methods[idx]["commitId"] != snapshot_commit:
            new_idx = idx
            break

    num_of_val_methods = (num_of_methods - new_idx) // 2
    val_methods = added_methods[new_idx : new_idx + num_of_val_methods]
    test_methods = added_methods[new_idx + num_of_val_methods :]
    write_classes(val_methods, val_path)
    write_classes(test_methods, test_path)

    return dataset_dir


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_names", type=str, help="A list of projects preprocessed via CommentUpdater")
    arg_parser.add_argument("train_part", type=float, help="Fraction of train part")

    args = arg_parser.parse_args()

    setup_comment_updater()

    with open(args.project_names, "r") as projects:
        for project in projects:
            split_dataset(project.strip(), args.train_part)
