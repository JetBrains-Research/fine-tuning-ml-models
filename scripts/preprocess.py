import os
import tempfile
from argparse import ArgumentParser
from shutil import copytree

from .load_tools import setup_psiminer
from .utils import PSIMINER_DIR, PREPROCESSED_DATASETS_DIR


def fix_naming(dataset_path: str) -> None:
    """Remove useless datasets"""

    old_name = os.path.join(dataset_path, "result.c2s")
    new_name = os.path.join(dataset_path, "java-med-psi-no-types.test.c2s")
    os.rename(old_name, new_name)


def preprocess(project_path: str) -> None:
    """Transform project into test data for code2seq via psiminer"""

    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = os.path.join(PREPROCESSED_DATASETS_DIR, project_name, "java-med-psi-no-types")
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        new_path = os.path.join(tmp, "test", project_name)
        copytree(project_path, new_path)
        cmd = f'bash {PSIMINER_DIR}/psiminer.sh "{tmp}" "{dataset_path}" configs/psiminer_v2_code2seq_config.json'
        os.system(cmd)
    fix_naming(dataset_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to project's files to be preprocessed")

    args = arg_parser.parse_args()

    setup_psiminer()
    preprocess(args.project)
