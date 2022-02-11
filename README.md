# Fine-tuning for Code2Seq and TreeLSTM

Framework for fine-tuning Code2Seq and TreeLSTM models on data extracted from Git history of Java projects

## Before experiments

Create conda environment and activate it:

```shell
conda env create --name <your-env-name> -f ft-env.yml
conda activate <your-env-name>
```

## Preprocessing

1. Prepare file with `.git` links to projects separated via `'\n'`
2. Path this file to script `experiments/repos_to_model_input`, also you need to specify a model type and percentage of
   methods, which should be considered as a train part of dataset
3. Resulting datasets can be found in folder `datasets`, source code partitioned in train/test/val and data mined from
   git history are in `extracted_methods` dir and projects themselves are in `cloned_repos` directory

Example run:

```shell
python -m experiments.repos_to_model_input links.txt code2seq 0.8
```

## Fine-tuning

1. After preprocessing, write names of projects from `preprocessed` folder to some text file
2. Pass name of this file as argument to `experiments/fine_tune_and_calc_results.py` script, also you need to specify a
   model type and path to it
3. Results for each project can be found in separate folders in `results` folder. Metrics and method names (target and
   predicted) are saved for three models: trained from scratch, original and fine-tuned
4. Fine-tuned and trained from scratch models with timestamps can be found at `models/fine-tuning-experiments`

Example run:

```shell
python -m experiments.fine_tune_and_calc_results names.txt code2seq models/code2seq.ckpt
```

**NB** Parts of these pipelines can be run separately, for more info check `scripts` folder

## Summarizing results

1. You can summarize metrics from `results` folder into some plots and mean metrics
   via `experiments/summarize_results.py`

Example run:

```shell
python -m experiments.summarize_results
```