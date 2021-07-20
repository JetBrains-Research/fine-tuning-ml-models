from argparse import ArgumentParser
import os
from typing import Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from .load_tools import setup_code2seq

setup_code2seq()

from code2seq.dataset import PathContextDataModule, TypedPathContextDataModule
from code2seq.model import Code2Seq, Code2Class, TypedCode2Seq
from code2seq.utils.callback import UploadCheckpointCallback, PrintEpochResultCallback
from code2seq.utils.vocabulary import Vocabulary


def load_code2seq(
        checkpoint_path: str, config: DictConfig, vocabulary: Vocabulary
) -> Tuple[Code2Seq, PathContextDataModule]:
    model = Code2Seq.load_from_checkpoint(checkpoint_path=checkpoint_path)
    data_module = PathContextDataModule(config, vocabulary)
    return model, data_module


def load_code2class(
        checkpoint_path: str, config: DictConfig, vocabulary: Vocabulary
) -> Tuple[Code2Class, PathContextDataModule]:
    model = Code2Class.load_from_checkpoint(checkpoint_path=checkpoint_path)
    data_module = PathContextDataModule(config, vocabulary)
    return model, data_module


def load_typed_code2seq(
        checkpoint_path: str, config: DictConfig, vocabulary: Vocabulary
) -> Tuple[TypedCode2Seq, TypedPathContextDataModule]:
    model = TypedCode2Seq.load_from_checkpoint(checkpoint_path=checkpoint_path)
    data_module = TypedPathContextDataModule(config, vocabulary)
    return model, data_module


KNOWN_MODELS = {"code2seq": load_code2seq, "code2class": load_code2class, "typed-code2seq": load_typed_code2seq}


def fine_tune(dataset_path: str, model_path: str, folds_number: int) -> None:
    project_name = os.path.basename(os.path.normpath(dataset_path))

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]["config"]
    vocabulary = checkpoint["hyper_parameters"]["vocabulary"]
    config.data_folder = dataset_path

    if config.name not in KNOWN_MODELS:
        print(f"Unknown model {config.name}, try one of {' '.join(KNOWN_MODELS.keys())}")
        return

    model, data_module = KNOWN_MODELS[config.name](model_path, config, vocabulary)

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("models", "fine_tuned", project_name),
        period=config.save_every_epoch,
        save_top_k=-1,
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

    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")
    arg_parser.add_argument("folds", type=int, help="Number of folds for k-fold cross-validation")

    args = arg_parser.parse_args()
    print(fine_tune(args.project, args.model, args.folds))
