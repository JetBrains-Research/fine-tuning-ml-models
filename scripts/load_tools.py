import git
import os
from .utils import PSIMINER_DIR, COMMENT_UPDATER_DIR, PSIMINER_COMMIT_ID, RunInDir


def setup_psiminer() -> None:
    """Load psiminer if needed"""

    if not os.path.exists(PSIMINER_DIR):
        link = "https://github.com/JetBrains-Research/psiminer.git"
        repo = git.Repo.clone_from(link, PSIMINER_DIR, multi_options=["-b master"])
        repo.head.reset(PSIMINER_COMMIT_ID, index=True, working_tree=True)

        with RunInDir(PSIMINER_DIR):
            os.system("./gradlew clean build")


def setup_comment_updater() -> None:
    """Load CommentUpdater if needed"""

    if not os.path.exists(COMMENT_UPDATER_DIR):
        link = "https://github.com/egor-bogomolov/MethodUpdateMiner.git"
        git.Repo.clone_from(link, COMMENT_UPDATER_DIR, multi_options=["--depth 1 -b dev-postprocessing"])


if __name__ == "__main__":
    setup_psiminer()
    setup_comment_updater()
