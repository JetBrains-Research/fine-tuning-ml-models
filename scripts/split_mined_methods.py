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


def split_dataset(project_name: str, val_part: float, test_part: str) -> str:
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

    num_of_val_methods = int(len(added_methods) * val_part)
    num_of_test_methods = int(len(added_methods) * test_part)
    val_methods = added_methods[num_of_test_methods : num_of_test_methods + num_of_val_methods]
    test_methods = added_methods[:num_of_test_methods]
    write_classes(val_methods, val_path)
    write_classes(test_methods, test_path)

    last_commit_time = val_methods[-1]["commitTime"]
    snapshot_commit_id = ""
    for i in range(num_of_test_methods + num_of_val_methods, len(added_methods)):
        if added_methods[i]["commitTime"] > last_commit_time:
            snapshot_commit_id = added_methods[i]["commitId"]

    source_dir = os.path.join(CLONED_REPOS_DIR, project_name)
    repo = git.Repo(source_dir)
    repo.head.reset(snapshot_commit_id, index=True, working_tree=True)
    copytree(source_dir, train_path)

    return dataset_dir


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_name", type=str, help="A name of folder in extracted methods folder")
    arg_parser.add_argument("val_part", type=float, help="Fraction of validation part")
    arg_parser.add_argument("test_part", type=float, help="Fraction of test part")

    args = arg_parser.parse_args()

    setup_comment_updater()

    split_dataset(args.project_link, args.val_part, args.test_part)
