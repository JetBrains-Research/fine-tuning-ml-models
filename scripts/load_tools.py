import sys
import git
import os
from .utils import CODE2SEQ_DIR, PSIMINER_DIR, COMMENTUPDATER_DIR, RunInDir


def setup_psiminer() -> None:
    """Load psiminer if needed"""

    if not os.path.exists(PSIMINER_DIR):
        link = "https://github.com/JetBrains-Research/psiminer.git"
        git.Repo.clone_from(link, PSIMINER_DIR, multi_options=["--depth 1 -b psiminer_v2"])

        with RunInDir(PSIMINER_DIR):
            os.system("./gradlew clean build")


def add_path_code2seq() -> None:
    sys.path.append(CODE2SEQ_DIR)
    sys.path.append(os.path.join(CODE2SEQ_DIR, "code2seq"))


def setup_code2seq() -> None:
    """Load code2seq if needed and add it to path"""

    if not os.path.exists(CODE2SEQ_DIR):
        link = "https://github.com/JetBrains-Research/code2seq.git"
        git.Repo.clone_from(link, CODE2SEQ_DIR, multi_options=["--depth 1 -b test-results-serialization"])
    add_path_code2seq()


def setup_comment_updater() -> None:
    """Load CommentUpdater if needed"""

    if not os.path.exists(COMMENTUPDATER_DIR):
        link = "https://github.com/malodetz/CommentUpdater.git"
        git.Repo.clone_from(link, COMMENTUPDATER_DIR, multi_options=["--depth 1 -b dev-postprocessing"])


if __name__ == "__main__":
    setup_psiminer()
    setup_code2seq()
    setup_comment_updater()
