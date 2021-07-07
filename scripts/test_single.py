from argparse import ArgumentParser
from load_tools import setup_code2seq

setup_code2seq()

from сode2seq.code2seq.test import test


def get_only_metrics(results):
    """Turn dictionary of results into a list of metrics"""

    metrics_names = ["test/f1", "test/precision", "test/recall", "test/loss"]
    metrics = [results[name] for name in metrics_names]
    return metrics


def test_single(model_path: str, project_path: str):
    """Evaluate model"""

    results = test(model_path, project_path)
    return get_only_metrics(results[0])


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")

    args = arg_parser.parse_args()
    print(test_single(args.model, args.project))
