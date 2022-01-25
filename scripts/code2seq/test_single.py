import torch
from pytorch_lightning import Trainer, seed_everything

from .fine_tune import get_pretrained_model
from ..utils import CODE2SEQ_VOCABULARY


def get_only_metrics(results):
    """Turn dictionary of results into a list of metrics"""

    metrics_names = ["test/f1", "test/precision", "test/recall", "test/loss"]
    metrics = [results[name] for name in metrics_names]
    return metrics


def test_single(
    model_path: str,
    project_path: str,
    is_from_scratch_model: bool,
    output: str = None,
    vocabulary_path: str = CODE2SEQ_VOCABULARY,
):
    """Evaluate model"""

    model, data_module, config, vocabulary = get_pretrained_model(
        model_path, project_path, is_from_scratch_model, vocabulary_path
    )
    seed_everything(config.seed)

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    results = trainer.test(model=model, datamodule=data_module)

    if output is not None:
        with open(output, "w") as f:
            print(*results, file=f)

    return get_only_metrics(results[0])
