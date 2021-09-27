from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything

from .fine_tune import get_pretrained_model
from ..utils import TREELSTM_VOCABULARY


def test_single(model_path: str, project_path: str, output: str = None, vocabulary_path: str = TREELSTM_VOCABULARY):
    """Evaluate model"""

    model, data_module, config, vocabulary = get_pretrained_model(model_path, project_path, vocabulary_path)
    seed_everything(config.seed)

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    results = trainer.test(model=model, datamodule=data_module)

    if output is not None:
        with open(output, "w") as f:
            print(*results, file=f)

    return results[0]


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project", type=str, help="Path to preprocessed project dataset")
    arg_parser.add_argument("model", type=str, help="Path to model checkpoint to be evaluated")
    arg_parser.add_argument("vocabulary", type=str)

    args = arg_parser.parse_args()
    print(test_single(args.model, args.project, None, args.vocabulary))
