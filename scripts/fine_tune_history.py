from argparse import ArgumentParser
import os
import tempfile
from .utils import CLONED_REPOS_DIR, EXTRACTED_METHODS_DIR, COMMENT_UPDATER_DIR, COMMENT_UPDATER_CONFIG_DIR
from .load_tools import setup_comment_updater
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


def fine_tune_history(link: str, first_commit: str, second_commit: str):
    print("Cloning repo...")
    project_name = clone_repo(link)
    print("Cloned!")
    print("Running update mining...")
    run_comment_updater(project_name)
    print("Mining completed")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_link", type=str, help="A .git link to clone project with all history")
    arg_parser.add_argument("commit1", type=str, help="Hash of 1st separation commit")
    arg_parser.add_argument("commit2", type=str, help="Hash of 2nd separation commit")

    args = arg_parser.parse_args()

    setup_comment_updater()

    fine_tune_history(args.project_link, args.commit1, args.commit2)
