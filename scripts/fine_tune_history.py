from argparse import ArgumentParser
import os
import tempfile
import json
from typing import List, Dict
from shutil import copytree
from .utils import (
    CLONED_REPOS_DIR,
    EXTRACTED_METHODS_DIR,
    COMMENT_UPDATER_DIR,
    COMMENT_UPDATER_CONFIG_DIR,
    PREPROCESSED_DATASETS_DIR,
)
from .load_tools import setup_comment_updater
from .preprocess import preprocess_complete
from .fine_tune import train_and_test
import git


def clone_repo(link: str) -> str:
    project_name = link.split(".git")[0].split("/")[-1]
    cloned_path = os.path.join(CLONED_REPOS_DIR, project_name)
    git.Repo.clone_from(link, cloned_path)
    return project_name


def run_comment_updater(project_name: str) -> None:
    result_path = os.path.join(EXTRACTED_METHODS_DIR, project_name)
    os.makedirs(result_path)
    stats_path = os.path.join(result_path, "stats.json")
    open(stats_path, "w")
    script_path = os.path.join(COMMENT_UPDATER_DIR, "comment_update_miner.sh")
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.txt")
        f = open(input_path, "w")
        f.write(os.path.abspath(os.path.join(CLONED_REPOS_DIR, project_name)))
        f.close()

        cmd = f"bash {script_path} {input_path} {result_path} {COMMENT_UPDATER_CONFIG_DIR} {stats_path}"
        os.system(cmd)


def write_classes(methods_list: List[Dict[str, str]], folder: str) -> None:
    for i in range(len(methods_list)):
        filename = f"A{i}.java"
        with open(os.path.join(folder, filename), "w") as f:
            f.write(f'public class A{i} {"{"}\n')
            f.write(methods_list[i]["code"])
            f.write("}\n")


def split_dataset(project_name: str, val_part: float, test_part: str) -> str:
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


def fine_tune_history(link: str, val_part: float, test_part: str, model_path: str):
    print("Cloning repo...")
    project_name = clone_repo(link)
    print("Cloned!")

    print("Running update mining...")
    run_comment_updater(project_name)
    print("Mining completed!")

    print("Extracting added methods...")
    raw_dataset = split_dataset(project_name, val_part, test_part)
    print("Extracted!")

    print("Preprocessing raw java to .c2s...")
    preprocess_complete(raw_dataset)
    print("Preprocessing finished!")

    print("Model evaluating and trained...")
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    model_folder = os.path.join("models", "history_fine_tuned", project_name)
    model_path, metrics_before, metrics_after = train_and_test(dataset_path, model_path, model_folder)
    print("Finished!")
    print("_" * 30)
    print("Metrics before:", metrics_before)
    print("Metrics after:", metrics_after)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_link", type=str, help="A .git link to clone project with all history")
    arg_parser.add_argument("val_part", type=float, help="Fraction of validation part")
    arg_parser.add_argument("test_part", type=float, help="Fraction of test part")
    arg_parser.add_argument(
        "--model", type=str, help="Already trained model to be fine-tuned", default=None, required=False
    )

    args = arg_parser.parse_args()

    setup_comment_updater()

    fine_tune_history(args.project_link, args.val_part, args.test_part, args.model)
