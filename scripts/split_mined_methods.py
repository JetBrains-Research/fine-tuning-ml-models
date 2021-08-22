from argparse import ArgumentParser
import os
import json
from typing import List, Dict
from shutil import copytree
from .utils import CLONED_REPOS_DIR, EXTRACTED_METHODS_DIR
from .load_tools import setup_comment_updater
import git


def write_classes(methods_list: List[Dict[str, str]], folder: str) -> None:
    """Wrap methods into separate java classes and write them"""

    for i in range(len(methods_list)):
        filename = f"A{i}.java"
        with open(os.path.join(folder, filename), "w") as f:
            f.write(f'public class A{i} {"{"}\n')
            f.write(methods_list[i]["code"])
            f.write("}\n")


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
    arg_parser.add_argument("project_name", type=str, help="A name of folder in extracted methods folder")
    arg_parser.add_argument("train_part", type=float, help="Fraction of train part")

    args = arg_parser.parse_args()

    setup_comment_updater()

    split_dataset(args.project_name, args.train_part)
