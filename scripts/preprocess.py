import os
from argparse import ArgumentParser
from load_tools import setup_psiminer
import tempfile
from shutil import copytree, rmtree


def fix_naming(dataset_path: str) -> None:
    """"Remove useless datasets"""

    train = os.path.join(dataset_path, "java-med-psi.train.c2s")
    val = os.path.join(dataset_path, "java-med-psi.val.c2s")
    os.remove(train)
    os.remove(val)


def preprocess(project_path: str) -> None:
    """Transform project into test data for code2seq via psiminer"""

    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join("..", 'datasets', project_name, 'java-med-psi')
    tmp = tempfile.mkdtemp(dir='.')
    new_path = os.path.join(tmp, "test", project_name)
    copytree(project_path, new_path)
    cmd = f'bash ./psiminer/psiminer.sh "{tmp}" "{dataset_path}" psiminer_config.json'
    os.system(cmd)
    rmtree(tmp)
    fix_naming(dataset_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to project's files to be preprocessed")

    args = arg_parser.parse_args()

    setup_psiminer()
    preprocess(args.project)
