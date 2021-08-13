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
    if batch_size is not None:
        config.batch_size = batch_size
        config.test_batch_size = batch_size
    vocabulary = checkpoint["hyper_parameters"]["vocabulary"]
    model = Code2Seq.load_from_checkpoint(checkpoint_path=model_path)
    data_module = PathContextDataModule(config, vocabulary)
    return model, data_module, config, vocabulary


def train_and_test(dataset_path: str, model_path: str, model_folder: str) -> Tuple[str, Any, Any]:
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

    return checkpoint_callback.best_model_path, metrics_before, metrics_after


def fine_tune(dataset_path: str, model_path: str, folds_number: int):
    """Do k-fold cross-validation and compare quality metrics before and after fine-tuning"""

    project_name = os.path.basename(os.path.normpath(dataset_path))
    start_path = os.path.join(dataset_path, NO_TYPES_PATH)
    dataset = open(os.path.join(start_path, f"{NO_TYPES_PATH}.test.c2s"), "r")
    samples = dataset.readlines()
    folds_number += 1
    fold_size = len(samples) // folds_number

    with tempfile.TemporaryDirectory(dir=".") as tmp, open("results.txt", "w") as result_file:
        for i in range(folds_number - 1):
            preprocessed_path = os.path.join(tmp, str(i + 1))
            fold_path = os.path.join(preprocessed_path, NO_TYPES_PATH)
            os.makedirs(fold_path)
            copy(os.path.join(start_path, "nodes_vocabulary.csv"), fold_path)

            with open(os.path.join(fold_path, f"{NO_TYPES_PATH}.train.c2s"), "w+") as train:
                train.writelines(samples[: i * fold_size])
                train.writelines(samples[(i + 2) * fold_size :])

            with open(os.path.join(fold_path, f"{NO_TYPES_PATH}.val.c2s"), "w+") as val:
                val.writelines(samples[(i + 1) * fold_size : (i + 2) * fold_size])

            with open(os.path.join(fold_path, f"{NO_TYPES_PATH}.test.c2s"), "w+") as test:
                test.writelines(samples[i * fold_size : (i + 1) * fold_size])

            print(f"Fold #{i}:", file=result_file)

            tuned_model_folder = os.path.join("models", "fine_tuned", project_name, str(i))
            trained_model_path, metrics_before, metrics_after = train_and_test(
                preprocessed_path, model_path, tuned_model_folder
            )
            print("Metrics before:", metrics_before, file=result_file)
            print("Metrics after:", metrics_after, file=result_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")
    arg_parser.add_argument("folds", type=int, help="Number of folds for k-fold cross-validation")

    args = arg_parser.parse_args()
    fine_tune(args.project, args.model, args.folds)
