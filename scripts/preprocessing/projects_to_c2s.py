from argparse import ArgumentParser
import os

from .preprocess import run_psiminer
from scripts.utils import PSIMINER_CODE2SEQ_TOPIC_CONFIG


def process_separate(dataset_path: str) -> None:
    for projects_set in ["training", "validation", "test"]:
        projects_set_folder = os.path.join(dataset_path, projects_set)
        cnt = 0
        for name in os.listdir(projects_set_folder):
            input_path = os.path.join(projects_set_folder, name)
            output_path = os.path.join("separate_lines", projects_set, name)
            run_psiminer(input_path, output_path, PSIMINER_CODE2SEQ_TOPIC_CONFIG)
            cnt += 1
            if cnt == 5:
                break


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("dataset_path", type=str, help="Path to project sources, split into train/test/val")
    args = arg_parser.parse_args()

    process_separate(args.dataset_path)
