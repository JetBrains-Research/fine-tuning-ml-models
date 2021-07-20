from argparse import ArgumentParser
import os
import tempfile
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

from .test_single import test_single


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


def train_and_test(dataset_path: str, model_path: str, fold_idx: int) -> str:
    """Trains model and return a path to best checkpoint"""

    project_name = os.path.basename(os.path.normpath(dataset_path))

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]["config"]
    vocabulary = checkpoint["hyper_parameters"]["vocabulary"]
    config.data_folder = dataset_path

    if config.name not in KNOWN_MODELS:
        print(f"Unknown model {config.name}, try one of {' '.join(KNOWN_MODELS.keys())}")
        return ""

    model, data_module = KNOWN_MODELS[config.name](model_path, config, vocabulary)

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("models", "fine_tuned", project_name, str(fold_idx)),
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

    return checkpoint_callback.best_model_path


def fine_tune(dataset_path: str, model_path: str, folds_number: int):
    dataset = open(os.path.join(dataset_path, "java-med-psi-no-types", "java-med-psi-no-types.test.c2s"), "r")
    samples = dataset.readlines()
    fold_size = len(samples) // folds_number

    with tempfile.TemporaryDirectory(dir=".") as tmp, open("results.txt", "w") as result_file:
        for i in range(folds_number - 1):
            preprocessed_path = os.path.join(tmp, str(i + 1))
            fold_path = os.path.join(preprocessed_path, "java-med-psi-no-types")
            os.makedirs(fold_path)

            with open(os.path.join(fold_path, "java-med-psi-no-types.train.c2s"), "w+") as train:
                train.writelines(samples[: i * fold_size])
                train.writelines(samples[(i + 2) * fold_size:])

            with open(os.path.join(fold_path, "java-med-psi-no-types.val.c2s"), "w+") as val:
                val.writelines(samples[(i + 1) * fold_size: (i + 2) * fold_size])

            with open(os.path.join(fold_path, "java-med-psi-no-types.test.c2s"), "w+") as test:
                test.writelines(samples[i * fold_size: (i + 1) * fold_size])

            print(f"Fold #{i}:", file=result_file)
            print("Metrics before:", test_single(model_path, preprocessed_path), file=result_file)
            trained_model_path = train_and_test(preprocessed_path, model_path, i)
            print("Metrics after:", test_single(trained_model_path, preprocessed_path), file=result_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")
    arg_parser.add_argument("folds", type=int, help="Number of folds for k-fold cross-validation")

    args = arg_parser.parse_args()
    fine_tune(args.project, args.model, args.folds)
