import os

DEPENDENCIES_DIR = "dependencies"
CODE2SEQ_DIR = os.path.join(DEPENDENCIES_DIR, "code2seq_repo")
PSIMINER_DIR = os.path.join(DEPENDENCIES_DIR, "psiminer_repo")
NO_TYPES_PATH = "java-med-psi-no-types"
COMMENTUPDATER_DIR = os.path.join(DEPENDENCIES_DIR, "CommentUpdater")
CONFIGS_DIR = "configs"
PSIMINER_CONFIG = os.path.join(CONFIGS_DIR, "psiminer_v2_code2seq_config.json")


class RunInDir:
    def __init__(self, path_to_dir: str) -> None:
        self.cwd = os.getcwd()
        self.path_to_dir = path_to_dir

    def __enter__(self) -> None:
        os.chdir(self.path_to_dir)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        os.chdir(self.cwd)
