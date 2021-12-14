from argparse import ArgumentParser
from typing import Tuple, Any, Optional

import torch
import dgl
from os.path import join
from commode_utils.callbacks import PrintEpochResultCallback
from commode_utils.vocabulary import build_from_scratch
from omegaconf import OmegaConf, DictConfig

from embeddings_for_trees.data.jsonl_data_module import JsonlASTDatamodule
from embeddings_for_trees.data.vocabulary import Vocabulary
from embeddings_for_trees.models import TreeLSTM2Seq
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from scripts.utils import TREELSTM_CONFIG, TREELSTM_VOCABULARY


class CustomVocabularyDataModule(JsonlASTDatamodule):
    def __init__(self, data_dir: str, config: DictConfig, vocabulary_path: str = None):
        self._vocabulary_path = vocabulary_path
        super().__init__(config, data_dir)

    def setup_vocabulary(self):
        if self._vocabulary_path is None:
            print("Can't find vocabulary, building")
            build_from_scratch(join(self._data_folder, f"{self._train}.jsonl"), Vocabulary)
            vocabulary_path = join(self._data_folder, Vocabulary.vocab_filename)
        else:
            vocabulary_path = self._vocabulary_path
        return Vocabulary(vocabulary_path, self._config.labels_count, self._config.tokens_count)


def get_config_data_module_vocabulary(dataset_path: str, is_from_scratch: bool, vocabulary_path: str = None):
    config = DictConfig(OmegaConf.load(TREELSTM_CONFIG))
    config.data_folder = dataset_path
    if is_from_scratch:
        config.data.labels_count = None
        config.data.tokens_count = None

    seed_everything(config.seed)
    dgl.seed(config.seed)

    data_module = CustomVocabularyDataModule(config.data_folder, config.data, vocabulary_path)
    data_module.setup()

    return config, data_module, data_module.vocabulary


def get_untrained_model(dataset_path: str):
    config, data_module, vocabulary = get_config_data_module_vocabulary(dataset_path, True)

    model = TreeLSTM2Seq(config.model, config.optimizer, data_module.vocabulary, config.train.teacher_forcing)

    return model, data_module, config, vocabulary


def get_pretrained_model(model_path: str, dataset_path: str, is_from_scratch: bool,
                         vocabulary_path: Optional[str] = TREELSTM_VOCABULARY):
    if vocabulary_path is None:
        vocabulary_path = TREELSTM_VOCABULARY

    config, data_module, vocabulary = get_config_data_module_vocabulary(dataset_path, is_from_scratch, vocabulary_path)

    model = TreeLSTM2Seq.load_from_checkpoint(model_path, map_location=torch.device("cpu"))

    return model, data_module, config, vocabulary


def train_and_test(dataset_path: str, model_folder: str, model_path: str = None) -> Tuple[str, Any, Any]:
    """Trains model and return a path to best checkpoint"""

    if model_path is not None:
        model, data_module, config, vocabulary = get_pretrained_model(model_path, dataset_path, is_from_scratch=False)
    else:
        model, data_module, config, vocabulary = get_untrained_model(dataset_path)

    params = config.train
    dgl.seed(config.seed)

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        every_n_epochs=params.save_every_epoch,
        monitor="val/loss",
        save_top_k=1,
    )

    # define other callbacks
    early_stopping_callback = EarlyStopping(patience=params.patience, monitor="val/loss", verbose=True, mode="min")
    print_epoch_result_callback = PrintEpochResultCallback(after_test=False)
    lr_logger = LearningRateMonitor("step")

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(
        max_epochs=params.n_epochs,
        gradient_clip_val=params.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=params.val_every_epoch,
        log_every_n_steps=params.log_every_n_steps,
        gpus=gpu,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        callbacks=[
            lr_logger,
            early_stopping_callback,
            checkpoint_callback,
            print_epoch_result_callback,
        ],
    )

    metrics_before = trainer.test(model=model, datamodule=data_module)
    trainer.fit(model=model, datamodule=data_module)
    metrics_after = trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=data_module)

    print("_" * 30)
    print("Metrics before:", metrics_before)
    print("Metrics after:", metrics_after)
    print("_" * 30)

    return checkpoint_callback.best_model_path, metrics_before, metrics_after


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("models_folder", type=str, help="Path to save best model checkpoint")
    arg_parser.add_argument(
        "--model", type=str, help="Already trained model to be fine-tuned", default=None, required=False
    )

    args = arg_parser.parse_args()
    train_and_test(args.project, args.models_folder, args.model)
