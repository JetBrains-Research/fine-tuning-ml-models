from argparse import ArgumentParser
import os
import tempfile
from shutil import copy
from typing import Tuple, Any

import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from .load_tools import setup_code2seq
from .utils import NO_TYPES_PATH, CODE2SEQ_CONFIG

setup_code2seq()

from code2seq.dataset import PathContextDataModule, TypedPathContextDataModule
from code2seq.model import Code2Seq, Code2Class, TypedCode2Seq
from code2seq.utils.callback import UploadCheckpointCallback, PrintEpochResultCallback
from code2seq.utils.vocabulary import Vocabulary
from code2seq.preprocessing.build_vocabulary import preprocess


def get_untrained_model(dataset_path: str):
    config = DictConfig(OmegaConf.load(CODE2SEQ_CONFIG))
    config.data_folder = dataset_path
    preprocess(config)
    vocabulary = Vocabulary.load_vocabulary(
        os.path.join(config.data_folder, config.dataset.name, config.vocabulary_name)
    )
    model = Code2Seq(config, vocabulary)
    data_module = PathContextDataModule(config, vocabulary)
    return model, data_module, config, vocabulary


def get_pretrained_model(model_path: str, dataset_path: str, batch_size: int = None):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]["config"]
    config.data_folder = dataset_path
    vocabulary = checkpoint["hyper_parameters"]["vocabulary"]
    model = Code2Seq.load_from_checkpoint(checkpoint_path=model_path)
    data_module = PathContextDataModule(config, vocabulary)
    return model, data_module, config, vocabulary


def train_and_test(dataset_path: str, model_folder: str, model_path: str = None) -> Tuple[str, Any, Any]:
    """Trains model and return a path to best checkpoint"""

    if model_path is not None:
        model, data_module, config, vocabulary = get_pretrained_model(model_path, dataset_path)
    else:
        model, data_module, config, vocabulary = get_untrained_model(dataset_path)

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        period=config.save_every_epoch,
        monitor="val_loss",
        save_top_k=1,
    )

    # define other callbacks
    early_stopping_callback = EarlyStopping(
        patience=config.hyper_parameters.patience, monitor="val_loss", verbose=True, mode="min"
    )
    print_epoch_result_callback = PrintEpochResultCallback("train", "val")
    lr_logger = LearningRateMonitor("step")

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.val_every_epoch,
        log_every_n_steps=config.log_every_epoch,
        gpus=gpu,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        callbacks=[
            checkpoint_callback,
            lr_logger,
            early_stopping_callback,
            print_epoch_result_callback,
        ],
        resume_from_checkpoint=model_path,
    )

    metrics_before = trainer.test(model=model, datamodule=data_module)
    trainer.fit(model=model, datamodule=data_module)
    metrics_after = trainer.test()

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
