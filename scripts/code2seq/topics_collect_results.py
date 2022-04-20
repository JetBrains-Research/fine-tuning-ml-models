import json
from argparse import ArgumentParser
import os
from ..utils import PREPROCESSED_DATASETS_DIR, RESULTS_DIR
from .restore_metrics import restore_c2s_metrics


def process_projects(data_path: str, topics_number: int) -> None:
    with open(data_path, "r") as f:
        data = json.load(f)[str(topics_number)]
    project_topics = data["test"]

    for project in project_topics:
        name = project.split("__")[1]
        dataset = os.path.join(PREPROCESSED_DATASETS_DIR, name)
        if not os.path.exists(dataset):
            continue

        model_dirs = os.path.join("models", "fine_tuning_experiments")
        model_dir = ""
        for model in os.listdir(model_dirs):
            m = model.split("_")
            if m[0] == str(topics_number) and m[1] == str(project_topics[project]):
                model_dir = os.path.join("models", "fine_tuning_experiments", model)

        if model_dir == "":
            raise ValueError("Topic model not found")

        vocabulary = os.path.join(PREPROCESSED_DATASETS_DIR, f"{topics_number}_{project_topics[project]}",
                                  "vocabulary.pkl")
        output_path = os.path.join(RESULTS_DIR, project)
        restore_c2s_metrics(dataset, model_dir, vocabulary, output_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("topics_distribution_path", type=str, help="Path to file with info about project topics")
    arg_parser.add_argument("number_of_topics", type=int, help="Number of project topics")
    args = arg_parser.parse_args()

    process_projects(args.topics_distribution_path, args.number_of_topics)
