from argparse import ArgumentParser
import os
import tempfile
from scripts.utils import (
    CLONED_REPOS_DIR,
    EXTRACTED_METHODS_DIR,
    MINER_DIR,
)


def run_miner(project_name: str) -> None:
    result_path = os.path.join(EXTRACTED_METHODS_DIR, project_name)
    os.makedirs(result_path)
    stats_path = os.path.join(result_path, "stats.json")
    open(stats_path, "w")
    script_path = os.path.join(MINER_DIR, "run_miner.sh")
    with tempfile.TemporaryDirectory(dir="../..") as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.txt")
        f = open(input_path, "w")
        f.write(os.path.abspath(os.path.join(CLONED_REPOS_DIR, project_name)))
        f.close()

        cmd = f"bash {script_path} {input_path} {result_path} {stats_path}"
        os.system(cmd)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_names", type=str, help="File with names of projects in cloned_repos folder")

    args = arg_parser.parse_args()

    with open(args.project_names, "r") as names:
        for name in names:
            run_miner(name.strip())
