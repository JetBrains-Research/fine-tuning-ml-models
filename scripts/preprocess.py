import os
from argparse import ArgumentParser
from shutil import copytree, rmtree
from git import Repo


def download_psiminer():
    link = "https://github.com/JetBrains-Research/psiminer.git"
    Repo.clone_from(link, "./psiminer")
    os.system("./psiminer/gradlew clean build")


def fix_naming(dataset_path: str):
    train = dataset_path + "java-med-psi.train.c2s"
    val = dataset_path + "java-med-psi.val.c2s"
    os.remove(train)
    os.remove(val)


def preprocess(project_path: str):
    if not os.path.isdir("./psiminer"):
        download_psiminer()
    project_name = os.path.basename(os.path.normpath(project_path))
    dataset_path = "../datasets/" + project_name + "/java-med-psi/"
    new_path = "temporary/test/" + project_name
    copytree(project_path, new_path)
    cmd = 'bash ./psiminer/psiminer.sh "{}" "{}" psiminer_config.json'.format("./temporary", dataset_path)
    os.system(cmd)
    rmtree("temporary")
    fix_naming(dataset_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str)

    args = arg_parser.parse_args()
    preprocess(args.project)
