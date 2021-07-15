[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![CI](https://github.com/JetBrains-Research/fine-tuning-ml-models/actions/workflows/ubuntu-python.yml/badge.svg?branch=test-all)

# Fine-tuning-ml-models

## Preprocessing

To extract data from PSI trees of Java project via [psiminer](https://github.com/JetBrains-Research/psiminer), use this
command

```console
$ python -m scripts.prepocess <path-to-project-folder>
```

Resulting .c2s file with samples and vocabulary table are written to the [`datasets`](datasets) directory, the name of
subdirectory name is same to the project's.

You can change extraction parameters (e.g., type preserving) by
modifying [`configs/psiminer_config.json`](configs/psiminer_config.json).

## Evaluating model on single project

To calculate quality metrics on a preprocessed project via [code2seq](https://github.com/JetBrains-Research/code2seq),
use this command

```console
$ python -m scripts.test_single <path-to-preprocessed-project> <path-to-model-checkpoint>
```

Function ``test_single`` returns the list of calculated metrics

## Evaluating model on all projects

```console
$ python -m scripts.test_single <path-to-preprocessed-raw-Java-dataset> <path-to-model-checkpoint> <path-to-results-storage-folder>
```

Automatically does preprocessing and evaluating, produces ``results.csv`` file, where all metrics and project names
stored with header.



