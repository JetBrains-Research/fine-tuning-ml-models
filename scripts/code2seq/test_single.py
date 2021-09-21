from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer

from fine_tune import get_pretrained_model


def get_only_metrics(results):
    """Turn dictionary of results into a list of metrics"""

    metrics_names = ["test/f1", "test/precision", "test/recall", "test/loss"]
    metrics = [results[name] for name in metrics_names]
    return metrics


def test_single(model_path: str, project_path: str):
    """Evaluate model"""

    model, data_module, config, vocabulary = get_pretrained_model(model_path, project_path)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    results = trainer.test(model, datamodule=data_module)

    return get_only_metrics(results[0])


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")

    args = arg_parser.parse_args()
    print(test_single(args.model, args.project))
