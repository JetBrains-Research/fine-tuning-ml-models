from argparse import ArgumentParser
from scripts.clone_repo import clone_repo
from scripts.mine_method_updates import run_comment_updater
from scripts.split_mined_methods import split_dataset
from scripts.preprocess import preprocess_complete
import os
from scripts.utils import PREPROCESSED_DATASETS_DIR


def process_single_repo(link: str, val_part: float, test_part: str) -> str:
    """Preprocess one repo into code2seq dataset"""

    print("Cloning repo", link)
    project_name = clone_repo(link)

    print("Running update mining...")
    run_comment_updater(project_name)
    print("Mining completed!")

    print("Extracting added methods...")
    raw_dataset = split_dataset(project_name, val_part, test_part)
    print("Extracted!")

    print("Preprocessing raw java to .c2s...")
    preprocess_complete(raw_dataset)
    print("Preprocessing finished!")

    return project_name


def process_many_repos(filename: str, val_part: float, test_part: str) -> None:
    """Start preprocessing for each repo and save its destination"""

    with open(filename, "r") as links_file, open(
        os.path.join(PREPROCESSED_DATASETS_DIR, "project_names.txt"), "w"
    ) as projects_file:
        for link in links_file:
            project = process_single_repo(link.strip(), val_part, test_part)
            projects_file.write(project + "\n")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "links_file", type=str, help="A path to file with .git link to clone project with all history"
    )
    arg_parser.add_argument("val_part", type=float, help="Fraction of validation part")
    arg_parser.add_argument("test_part", type=float, help="Fraction of test part")
    args = arg_parser.parse_args()

    process_many_repos(args.links_file, args.val_part, args.test_part)
