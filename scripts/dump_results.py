from argparse import ArgumentParser

import torch

from .load_tools import setup_code2seq

setup_code2seq()

from typing import List, Dict, Tuple
from code2seq.utils.vocabulary import Vocabulary, SOS, EOS, PAD
from .fine_tune import get_pretrained_model


def decode(sample: torch.Tensor, id_to_label: Dict[int, str], ignore_index: List[int]) -> List[str]:
    return [id_to_label[i.item()] for i in sample if i.item() not in ignore_index]


def extract(
    checkpoint_path: str, data_folder: str, batch_size: int = None, result_file: str = None
) -> List[Tuple[str, str]]:
    model, datamodule, config, vocabulary = get_pretrained_model(checkpoint_path, data_folder, batch_size)
    model.eval()

    id_to_label = {v: k for k, v in vocabulary.label_to_id.items()}
    ignore_index = [vocabulary.label_to_id[i] for i in [SOS, EOS, PAD]]

    if result_file is not None:
        f = open(result_file, "w")
    else:
        f = None
    results = []
    for batch in datamodule.test_dataloader():
        logits = model(batch.contexts, batch.contexts_per_label, batch.labels.shape[0])
        predictions = logits.argmax(-1)
        for y_true, y_pred in zip(batch.labels.t(), predictions.t()):
            y_true_decode = "|".join(decode(y_true, id_to_label, ignore_index))
            y_pred_decode = "|".join(decode(y_pred, id_to_label, ignore_index))
            results.append((y_true_decode, y_pred_decode))
            if f is not None:
                print(y_true_decode, y_pred_decode, file=f)

    return results


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("data_folder", type=str, default=None)
    arg_parser.add_argument("batch_size", type=int, default=None)

    args = arg_parser.parse_args()

    for item in extract(args.checkpoint, args.data_folder, args.batch_size):
        print(item)
