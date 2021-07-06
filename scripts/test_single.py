from argparse import ArgumentParser
import sys

from Code2seq.code2seq.test import test


def test_single(model, project):
    print(test(model, project))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str)
    arg_parser.add_argument("model", type=str)

    args = arg_parser.parse_args()
    test_single(args.model, args.project)
