import sys
import git
import os


def setup_psiminer() -> None:
    """Load psiminer if needed"""

    if not os.path.isdir("./psiminer"):
        link = "https://github.com/JetBrains-Research/psiminer.git"
        git.Repo.clone_from(link, "./psiminer", multi_options=["--depth 1 -b master"])

        os.chdir("psiminer")
        os.system("./gradlew clean build")
        os.chdir("..")


def add_path_code2seq() -> None:
    sys.path.append("./code2seq/code2seq")
    sys.path.append("./code2seq/code2seq/dataset")


def setup_code2seq() -> None:
    """Load code2seq if needed and add it to path"""

    if not os.path.isdir("code2seq"):
        link = "https://github.com/JetBrains-Research/code2seq.git"
        git.Repo.clone_from(link, "./code2seq", multi_options=["--depth 1 -b test-results-serialization"])
    add_path_code2seq()
    print(sys.path)


if __name__ == "__main__":
    setup_psiminer()
    setup_code2seq()
