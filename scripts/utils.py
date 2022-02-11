import os

DEPENDENCIES_DIR = "dependencies"
PREPROCESSED_DATASETS_DIR = "datasets"
PSIMINER_DIR = os.path.join(DEPENDENCIES_DIR, "psiminer")
NO_TYPES_PATH = "java-med-psi-no-types"
MINER_DIR = os.path.join(DEPENDENCIES_DIR, "Miner")
CONFIGS_DIR = "configs"
PSIMINER_CODE2SEQ_CONFIG = os.path.join(CONFIGS_DIR, "psiminer_code2seq_config.json")
PSIMINER_TREELSTM_CONFIG = os.path.join(CONFIGS_DIR, "psiminer_treelstm_config.json")
CODE2SEQ_CONFIG = os.path.join(CONFIGS_DIR, "code2seq_config.yaml")
CODE2SEQ_VOCABULARY = os.path.join(CONFIGS_DIR, "code2seq_vocabulary.pkl")
TREELSTM_CONFIG = os.path.join(CONFIGS_DIR, "treelstm_config.yaml")
TREELSTM_VOCABULARY = os.path.join(CONFIGS_DIR, "treelstm_vocabulary.pkl")
CLONED_REPOS_DIR = "cloned_repos"
EXTRACTED_METHODS_DIR = "extracted_methods"
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
