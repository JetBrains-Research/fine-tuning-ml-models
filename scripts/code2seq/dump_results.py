from argparse import ArgumentParser

import torch

from typing import List, Dict, Tuple
from .fine_tune import get_pretrained_model


def decode(sample: torch.Tensor, id_to_label: Dict[int, str], ignore_index: List[int]) -> List[str]:
    return [id_to_label[i.item()] for i in sample if i.item() not in ignore_index]


def extract(
        checkpoint_path: str, data_folder: str, is_from_scratch_model: bool, vocabulary_path: str = None,
        result_file: str = None
) -> List[Tuple[str, str]]:
    model, datamodule, config, vocabulary = get_pretrained_model(checkpoint_path, data_folder, is_from_scratch_model,
                                                                 vocabulary_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_to_label = {v: k for k, v in vocabulary.label_to_id.items()}
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    ignore_index = [vocabulary.label_to_id[i] for i in [SOS, EOS, PAD]]

    if result_file is not None:
        f = open(result_file, "w")
        serialization_needed = True
    else:
        serialization_needed = False
    results = []
    for batch in datamodule.test_dataloader():
        datamodule.transfer_batch_to_device(batch, device, 0)
        logits, _ = model.logits_from_batch(batch, None)
        with torch.no_grad():
            predictions = logits.argmax(-1)
        for y_true, y_pred in zip(batch.labels.t(), predictions.t()):
            y_true_decode = decode(y_true, id_to_label, ignore_index)
            y_pred_decode = decode(y_pred, id_to_label, ignore_index)
            y_true_decode = [x for x in y_true_decode if x != UNK]
            if len(y_true_decode) == 0:
                continue
            y_pred_decode = [x for x in y_pred_decode if x != UNK]

            y_true_decode = "|".join(y_true_decode)
            y_pred_decode = "|".join(y_pred_decode)
            results.append((y_true_decode, y_pred_decode))
            if serialization_needed:
                print(y_true_decode, y_pred_decode, file=f)

    return results


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("data_folder", type=str, default=None)
    arg_parser.add_argument("output", type=str, default=None)
    arg_parser.add_argument("--vocabulary", type=str, default=None, required=False)

    args = arg_parser.parse_args()

    if args.output is None:
        for item in extract(args.checkpoint, args.data_folder):
            print(item)
    else:
        extract(args.checkpoint, args.data_folder, result_file=args.output, vocabulary_path=args.vocabulary)
