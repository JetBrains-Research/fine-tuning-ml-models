from argparse import ArgumentParser
import os

from scripts.clone_repo import clone_repo
from scripts.mine_method_updates import run_comment_updater
from scripts.split_mined_methods import split_dataset
from .utils import PREPROCESSED_DATASETS_DIR
from .load_tools import setup_comment_updater
from .preprocess import preprocess_complete
from .fine_tune import train_and_test


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
