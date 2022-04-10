from argparse import ArgumentParser
import os
import json
from scripts.utils import PREPROCESSED_DATASETS_DIR


def unite_by_topics(data_path: str, topics_number: int) -> None:
    with open(data_path, "r") as f:
        data = json.load(f)[str(topics_number)]
    for i in range(topics_number):
        output_dir = os.path.join(PREPROCESSED_DATASETS_DIR, f"{topics_number}_{i}")
        os.mkdir(output_dir)
        for projects_set, short_name in zip(["training", "validation", "test"], ["train", "val", "test"]):
            projects = [k for k, v in data[projects_set].items() if v == i]
            with open(os.path.join(output_dir, f"{short_name}.c2s"), "w") as f:
                for project in projects:
                    with open(os.path.join("separate_lines", projects_set, project, "result.c2s"), "r") as f1:
                        f.writelines(f1.readlines())


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("topics_distribution_path", type=str, help="Path to file with info about project topics")
    arg_parser.add_argument("number_of_topics", type=int, help="Number of project topics")
    args = arg_parser.parse_args()

    unite_by_topics(args.topics_distribution_path, args.number_of_topics)
