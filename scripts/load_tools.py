import sys

import git
import os


def setup_psiminer() -> None:
    """Load psiminer if needed"""

    if not os.path.isdir("./psiminer"):
        link = "https://github.com/JetBrains-Research/psiminer.git"
        git.Repo.clone_from(link, "./psiminer", multi_options=['--depth 1 -b master'])
        os.system("./psiminer/gradlew clean build")


def setup_code2seq() -> None:
    """Load code2seq if needed and add it to path"""

    if not os.path.isdir("сode2seq"):
        link = "https://github.com/JetBrains-Research/code2seq.git"
        git.Repo.clone_from(link, "./сode2seq", multi_options=['--depth 1 -b test-results-serialization'])
    sys.path.append("./сode2seq/code2seq")


if __name__ == "__main__":
    setup_psiminer()
    setup_code2seq()
