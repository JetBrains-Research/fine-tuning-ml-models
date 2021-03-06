from argparse import ArgumentParser

import nltk
from nltk import meteor
import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def prepare_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


def calc_chrf(target: str, prediction: str) -> float:
    chrf_metric = CHRF()
    result = chrf_metric.sentence_score(prediction, [target])
    return result.score


def calc_bleu(target: str, prediction: str) -> float:
    bleu_metric = BLEU(effective_order=True, smooth_method="add-k")
    result = bleu_metric.sentence_score(prediction, [target])
    return result.score


def calculate_metrics(samples) -> pd.DataFrame:
    prepare_nltk()

    result = pd.DataFrame()
    bleus = []
    meteors = []
    chrfs = []
    for sample in samples:
        if len(sample) != 2:
            bleus.append(0.0)
            meteors.append(0.0)
            chrfs.append(0.0)
            continue
        target, predicted = sample[0].split("|"), sample[1].split("|")
        target_sentence, predicted_sentence = " ".join(target), " ".join(predicted)

        bleu_val = calc_bleu(target_sentence, predicted_sentence)
        meteor_val = meteor([target_sentence], predicted_sentence)
        chrf_val = calc_chrf(target_sentence, predicted_sentence)

        bleus.append(bleu_val)
        meteors.append(meteor_val)
        chrfs.append(chrf_val)

    bleu_values = np.array(bleus)
    bleu_mean = np.mean(bleu_values)
    meteor_values = np.array(meteors)
    meteor_mean = np.mean(meteor_values)
    chrf_values = np.array(chrfs)
    chrf_mean = np.mean(chrf_values)

    result["bleu"] = bleu_values
    result["meteor"] = meteor_values
    result["chrf"] = chrf_values

    means = pd.DataFrame([[bleu_mean, meteor_mean, chrf_mean]], columns=["bleu", "meteor", "chrf"], index=["means"])
    result = pd.concat([result, means])

    return result


def calculate_and_dump_metrics(input_file: str, output_file: str):
    samples = []
    with open(input_file, "r") as f:
        for line in f:
            samples.append(tuple(line.strip().split()))
    metrics = calculate_metrics(samples)
    metrics.to_csv(output_file, index=True)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("samples", type=str, help="Path to file with samples")
    arg_parser.add_argument("output", type=str, help="Path to file to save metrics")

    args = arg_parser.parse_args()

    calculate_and_dump_metrics(args.samples, args.output)
