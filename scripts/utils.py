import os

DEPENDENCIES_DIR = "dependencies"
CODE2SEQ_DIR = os.path.join(DEPENDENCIES_DIR, "code2seq_repo")
PSIMINER_DIR = os.path.join(DEPENDENCIES_DIR, "psiminer_repo")
NO_TYPES_PATH = "java-med-psi-no-types"
COMMENT_UPDATER_DIR = os.path.join(DEPENDENCIES_DIR, "CommentUpdater")
CONFIGS_DIR = "configs"
PSIMINER_CONFIG = os.path.join(CONFIGS_DIR, "psiminer_v2_code2seq_config.json")
COMMENT_UPDATER_CONFIG_DIR = os.path.join(CONFIGS_DIR, "comment_updater_config")
CODE2SEQ_CONFIG = os.path.join(CONFIGS_DIR, "code2seq_config.yaml")
PREPROCESSED_DATASETS_DIR = "datasets"
CLONED_REPOS_DIR = "cloned_repos"
EXTRACTED_METHODS_DIR = "extracted_methods"
PSIMINER_COMMIT_ID = "3487e323e7527bbe8cf3d483cb89fc0e9531a635"
RESULTS_DIR = "results"
EXPERIMENT_MODEL_DIR = os.path.join("models", "fine_tuning_experiments")


class RunInDir:
    def __init__(self, path_to_dir: str) -> None:
        self.cwd = os.getcwd()
        self.path_to_dir = path_to_dir

    def __enter__(self) -> None:
        os.chdir(self.path_to_dir)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        os.chdir(self.cwd)
