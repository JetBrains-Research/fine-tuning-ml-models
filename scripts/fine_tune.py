from argparse import ArgumentParser
from typing import Tuple, Any

import torch
from commode_utils.callback import PrintEpochResultCallback
from omegaconf import DictConfig, OmegaConf
from code2seq.data.path_context_data_module import PathContextDataModule
from code2seq.model import Code2Seq
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from .utils import CODE2SEQ_CONFIG


def get_config_data_module_vocabulary(dataset_path: str):
    config = DictConfig(OmegaConf.load(CODE2SEQ_CONFIG))
    config.data_folder = dataset_path

    data_module = PathContextDataModule(config.data_folder, config.data)
    data_module.prepare_data()
    data_module.setup()

    return config, data_module, data_module.vocabulary


def get_untrained_model(dataset_path: str):
    config, data_module, vocabulary = get_config_data_module_vocabulary(dataset_path)

    model = Code2Seq(config.model, config.optimizer, data_module.vocabulary, config.train.teacher_forcing)

    return model, data_module, config, data_module.vocabulary


def get_pretrained_model(model_path: str, dataset_path: str, batch_size: int = None):
    config, data_module, vocabulary = get_config_data_module_vocabulary(dataset_path)

    model = Code2Seq.load_from_checkpoint(model_path, map_location=torch.device("cpu"))

    return model, data_module, config, vocabulary


def train_and_test(dataset_path: str, model_folder: str, model_path: str = None) -> Tuple[str, Any, Any]:
    """Trains model and return a path to best checkpoint"""

    if model_path is not None:
        model, data_module, config, vocabulary = get_pretrained_model(model_path, dataset_path)
    else:
        model, data_module, config, vocabulary = get_untrained_model(dataset_path)

    params = config.train

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        every_n_epochs=params.save_every_epoch,
        monitor="val_loss",
        save_top_k=1,
    )

    # define other callbacks
    early_stopping_callback = EarlyStopping(patience=params.patience, monitor="val/loss", verbose=True, mode="min")
    print_epoch_result_callback = PrintEpochResultCallback(after_train=True, after_validation=True)
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
