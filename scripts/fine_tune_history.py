from argparse import ArgumentParser
import os

from .utils import PREPROCESSED_DATASETS_DIR
from .load_tools import setup_comment_updater
from .fine_tune import train_and_test
from experiments.repos_to_code2seq import process_single_repo


def fine_tune_history(link: str, train_part: float, model_path: str):
    project_name = process_single_repo(link, train_part)

    print("Model evaluating and trained...")
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name)
    model_folder = os.path.join("models", "history_fine_tuned", project_name)
    train_and_test(dataset_path, model_folder, model_path)
    print("Finished!")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_link", type=str, help="A .git link to clone project with all history")
    arg_parser.add_argument("train_part", type=float, help="Fraction of train part")
    arg_parser.add_argument(
        "--model", type=str, help="Already trained model to be fine-tuned", default=None, required=False
    )

    args = arg_parser.parse_args()

    setup_comment_updater()

    fine_tune_history(args.project_link, args.train_part, args.model)
