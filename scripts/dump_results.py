from argparse import ArgumentParser

import torch

from .load_tools import setup_code2seq

setup_code2seq()

from typing import List, Dict
from code2seq.dataset import PathContextDataModule, TypedPathContextDataModule
from code2seq.model import Code2Seq, Code2Class, TypedCode2Seq
from code2seq.utils.callback import UploadCheckpointCallback, PrintEpochResultCallback
from code2seq.utils.vocabulary import Vocabulary, SOS, EOS, PAD
from code2seq.test import KNOWN_MODELS


def decode(sample: torch.Tensor, id_to_label: Dict[int, str], ignore_index: List[int]) -> List[str]:
    return [id_to_label[i.item()] for i in sample if i.item() not in ignore_index]


def extract(checkpoint_path: str, data_folder: str = None, batch_size: int = None) -> List[(str, str)]:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]["config"]
    vocabulary = checkpoint["hyper_parameters"]["vocabulary"]
    if data_folder is not None:
        config.data_folder = data_folder
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size

    datamodule = PathContextDataModule(config, vocabulary)
    model = Code2Seq.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    id_to_label = {v: k for k, v in vocabulary.label_to_id.items()}
    ignore_index = [vocabulary.label_to_id[i] for i in [SOS, EOS, PAD]]

    results = []
    for batch in datamodule.test_dataloader():
        logits = model(batch.contexts, batch.contexts_per_label, batch.labels.shape[0])
        predictions = logits.argmax(-1)
        for y_true, y_pred in zip(batch.labels.t(), predictions.t()):
            y_true_decode = "|".join(decode(y_true, id_to_label, ignore_index))
            y_pred_decode = "|".join(decode(y_pred, id_to_label, ignore_index))
            results.append((y_true_decode, y_pred_decode))

    return results


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("data_folder", type=str, default=None)
    arg_parser.add_argument("batch_size", type=int, default=None)

    args = arg_parser.parse_args()

    extract(args.checkpoint, args.data_folder, args.batch_size)
