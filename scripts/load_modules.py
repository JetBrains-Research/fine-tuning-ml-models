import sys

import git
import os


def download_psiminer():
    if not os.path.isdir("./psiminer"):
        link = "https://github.com/JetBrains-Research/psiminer.git"
        git.Repo.clone_from(link, "./psiminer")
        os.system("./psiminer/gradlew clean build")


def download_code2seq():
    if not os.path.isdir("Code2seq"):
        link = "https://github.com/JetBrains-Research/code2seq.git"
        repo = git.Repo.clone_from(link, "./Code2seq")
        repo.git.checkout("test-results-serialization")
    sys.path.append("./Code2seq/code2seq")


if __name__ == "__main__":
    download_psiminer()
    download_code2seq()
