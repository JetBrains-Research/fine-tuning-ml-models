from argparse import ArgumentParser

from .load_tools import setup_code2seq

setup_code2seq()

from code2seq.test import test


def test_single(model, project):
    """Evaluate model"""

    print(test(model, project))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")

    args = arg_parser.parse_args()
    test_single(args.model, args.project)
