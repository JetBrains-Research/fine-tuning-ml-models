import dgl
import torch
from pytorch_lightning import Trainer, seed_everything

from .fine_tune import get_pretrained_model
from ..utils import TREELSTM_VOCABULARY


def test_single(
    model_path: str,
    project_path: str,
    is_from_scratch_model: bool,
    output: str = None,
    vocabulary_path: str = TREELSTM_VOCABULARY,
):
    """Evaluate model"""

    model, data_module, config, vocabulary = get_pretrained_model(
        model_path, project_path, is_from_scratch_model, vocabulary_path
    )
    seed_everything(config.seed)
    dgl.seed(config.seed)

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    results = trainer.test(model=model, datamodule=data_module)

    if output is not None:
        with open(output, "w") as f:
            print(*results, file=f)

    return results[0]
