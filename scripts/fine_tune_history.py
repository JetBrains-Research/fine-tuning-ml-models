from argparse import ArgumentParser
import os
import tempfile
import json
from shutil import copytree
from .utils import CLONED_REPOS_DIR, EXTRACTED_METHODS_DIR, COMMENT_UPDATER_DIR, COMMENT_UPDATER_CONFIG_DIR, \
    PREPROCESSED_DATASETS_DIR
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
        print(os.path.abspath(os.path.join(CLONED_REPOS_DIR, project_name)), file=f)
        f.close()

        cmd = f"bash {script_path} {input_path} {result_path} {COMMENT_UPDATER_CONFIG_DIR} {stats_path}"
        os.system(cmd)


def write_classes(methods_list, folder: str) -> None:
    for i in range(len(methods_list)):
        filename = f"A{i}.java"
        with open(os.path.join(folder, filename), "w") as f:
            f.write("public class A" + str(i) + "{\n")
            f.write(methods_list[i])
            f.write("}\n")


def split_dataset(project_name: str, first_commit: str, second_commit: str) -> str:
    dataset_dir = os.path.join(EXTRACTED_METHODS_DIR, project_name)
    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "val", project_name)
    os.makedirs(val_path)
    test_path = os.path.join(dataset_dir, "test", project_name)
    os.makedirs(test_path)

    source_dir = os.path.join(CLONED_REPOS_DIR, project_name)
    repo = git.Repo(source_dir)
    repo.head.reset(first_commit, index=True, working_tree=True)
    copytree(source_dir, train_path)

    first_commit_time = int(repo.commit(first_commit).committed_date)
    second_commit_time = int(repo.commit(second_commit).committed_date)
    print(first_commit_time, second_commit_time)

    val_methods = []
    test_methods = []

    raw_samples = open(os.path.join(dataset_dir, f"{project_name}.jsonl"), "r")
    for sample in raw_samples.readlines():
        data = json.loads(sample)
        if data["isNew"]:
            time = int(data["commitTime"]) // 1000
            if first_commit_time < time <= second_commit_time:
                val_methods.append(data["newCode"])
            elif second_commit_time < time:
                test_methods.append(data["newCode"])

    write_classes(val_methods, val_path)
    write_classes(test_methods, test_path)

    return dataset_dir


def fine_tune_history(link: str, first_commit: str, second_commit: str):
    print("Cloning repo...")
    project_name = clone_repo(link)
    print("Cloned!")

    print("Running update mining...")
    run_comment_updater(project_name)
    print("Mining completed!")

    print("Extracting added methods...")
    raw_dataset = split_dataset(project_name, first_commit, second_commit)
    print("Extracted!")

    print("Preprocessing raw java to .c2s...")
    preprocess_complete(raw_dataset)
    print("Preprocessing finished!")

    print("Model evaluating and trained...")
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    model_path = os.path.join("models", "epoch=08-val_loss=14.9236.ckpt")
    model_folder = os.path.join("models", "history_fine_tuned", project_name)
    model_path, metrics_before, metrics_after = train_and_test(dataset_path, model_path, model_folder)
    print("Finished!")
    print("___________________________________________________________________________________________")
    print("Metrics before:", metrics_before)
    print("Metrics after:", metrics_after)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_link", type=str, help="A .git link to clone project with all history")
    arg_parser.add_argument("commit1", type=str, help="Hash of 1st separation commit")
    arg_parser.add_argument("commit2", type=str, help="Hash of 2nd separation commit")

    args = arg_parser.parse_args()

    setup_comment_updater()

    fine_tune_history(args.project_link, args.commit1, args.commit2)
