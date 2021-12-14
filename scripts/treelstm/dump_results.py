import torch
import dgl
import os
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics

from typing import List, Dict
from .fine_tune import get_pretrained_model


def decode(sample: torch.Tensor, id_to_label: Dict[int, str], ignore_index: List[int]) -> List[str]:
    return [id_to_label[i.item()] for i in sample if i.item() not in ignore_index]


def extract(
    checkpoint_path: str,
    data_folder: str,
    is_from_scratch_model: bool,
    vocabulary_path: str = None,
    result_file: str = None,
) -> ClassificationMetrics:
    model, datamodule, config, vocabulary = get_pretrained_model(
        checkpoint_path, data_folder, is_from_scratch_model, vocabulary_path
    )
    dgl.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_to_label = {v: k for k, v in vocabulary.label_to_id.items()}
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    ignore_index = [vocabulary.label_to_id[i] for i in [SOS, EOS, PAD]]

    metrics = SequentialF1Score(vocabulary.label_to_id[PAD], vocabulary.label_to_id[EOS])
    all_predictions = []
    for batch in datamodule.test_dataloader():
        datamodule.transfer_batch_to_device(batch, device, 0)
        labels, graph = batch
        labels.to(device)
        graph = graph.to(device)
        logits, _ = model(graph, labels.shape[0])
        with torch.no_grad():
            predictions = logits.argmax(-1)
        for y_true, y_pred in zip(batch.labels.t(), predictions.t()):
            y_pred_filtered = torch.Tensor([x for x in y_pred if id_to_label[x.item()] != UNK])
            all_predictions.append("|".join(decode(y_pred_filtered, id_to_label, ignore_index)))

            pred = torch.stack([y_pred_filtered]).t().to(device)
            target = torch.stack([y_true]).t().to(device)
            metrics.update(pred, target)
    all_targets = []
    with open(os.path.join(data_folder, "test.c2s")) as f:
        for line in f:
            x = line.split()[0]
            all_targets.append(x)
    if result_file is not None:
        with open(result_file, "w") as f:
            for (x, y) in zip(all_targets, all_predictions):
                print(x, y, file=f)

    return metrics.compute()
