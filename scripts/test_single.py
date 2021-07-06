import sys
from argparse import ArgumentParser
from load_modules import download_code2seq

download_code2seq()

from сode2seq.code2seq.test import test


def test_single(model, project):
    print(test(model, project))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str)
    arg_parser.add_argument("model", type=str)

    args = arg_parser.parse_args()
    print(sys.path)
    test_single(args.model, args.project)
