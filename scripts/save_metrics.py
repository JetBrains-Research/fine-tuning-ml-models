from argparse import ArgumentParser

import nltk
from nltk import bleu, meteor
from nltk.translate import chrf_score
import numpy as np
import pandas as pd


def prepare_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


def calculate_metrics(samples) -> pd.DataFrame:
    prepare_nltk()

    result = pd.DataFrame()
    bleus = []
    meteors = []
    chrfs = []
    for sample in samples:
        target, predicted = sample[0].split("|"), sample[1].split("|")
        target_sentence, predicted_sentence = " ".join(target), " ".join(predicted)

        bleu_val = bleu([target], predicted)
        meteor_val = meteor([target_sentence], predicted_sentence)
        chrf_val = chrf_score.sentence_chrf(target_sentence, predicted_sentence)

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


def calculate_and_dump_metrics(samples, output_file: str):
    metrics = calculate_metrics(samples)
    metrics.to_csv(output_file, index=True)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("samples", type=str, help="Path to file with samples")
    arg_parser.add_argument("output", type=str, help="Path to file to save metrics")

    args = arg_parser.parse_args()

    lines = []
    with open(args.samples, "r") as f:
        for line in f:
            lines.append(tuple(line.strip().split()))
    calculate_and_dump_metrics(lines, args.output)
