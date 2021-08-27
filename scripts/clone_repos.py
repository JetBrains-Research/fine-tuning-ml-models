from argparse import ArgumentParser
import os
from .utils import CLONED_REPOS_DIR
import git


def clone_repo(link: str) -> str:
    project_name = link.split(".git")[0].split("/")[-1]
    cloned_path = os.path.join(CLONED_REPOS_DIR, project_name)
    git.Repo.clone_from(link, cloned_path)
    return project_name


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_links", type=str, help="A file with .git links to clone project with all history")

    args = arg_parser.parse_args()
    with open(args.project_links, "r") as f:
        for link in f:
            clone_repo(link.strip())
