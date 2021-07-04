from argparse import ArgumentParser
import tempfile, os, time
from typing import Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from code2seq.dataset import PathContextDataModule, TypedPathContextDataModule
from code2seq.model import Code2Seq, Code2Class, TypedCode2Seq
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


def test(checkpoint_path: str, data_folder: str = None, batch_size: int = None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]["config"]
    vocabulary = checkpoint["hyper_parameters"]["vocabulary"]
    if data_folder is not None:
        config.data_folder = data_folder
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size

    if config.name not in KNOWN_MODELS:
        print(f"Unknown model {config.name}, try one of {' '.join(KNOWN_MODELS.keys())}")
        return
    model, data_module = KNOWN_MODELS[config.name](checkpoint_path, config, vocabulary)

    seed_everything(config.seed)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    return trainer.test(model, datamodule=data_module)


def preprocess(project_path: str, psiminer_path: str) -> str:
    temporary_dir = tempfile.TemporaryDirectory()
    preprocessed_path = temporary_dir.name + "/java-med-psi/"
    psiminer_path += "psiminer.sh"
    config_path = os.getcwd() + "/psiminer_config.json"
    print(psiminer_path)
    print(project_path)
    print(temporary_dir.name)
    print(config_path)
    os.chdir("/")
    os.system(
        "bash \"{}\" \"{}\" \"{}\" \"{}\"".format(psiminer_path, project_path, preprocessed_path, config_path))
    return temporary_dir.name


def test_single(project_path: str, model_path: str, psiminer_path: str):
    data_path = preprocess(project_path, psiminer_path)
    result = test(model_path, data_path)
    print(result)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str)
    arg_parser.add_argument("model", type=str)
    arg_parser.add_argument("psiminer", type=str)

    args = arg_parser.parse_args()
    test_single(args.project, args.model, args.psiminer)
