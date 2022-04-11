import os
import pandas as pd

from scripts.utils import NODES_VOCABULARY


def fix_path(path: str) -> str:
    a = path.split(",")
    a[1] = "|".join(str(node_to_id[id_to_node[int(x)]]) for x in a[1].split("|"))
    return ",".join(a)


def process_sample(sample: str) -> str:
    paths = sample.split(" ")
    for i in range(1, len(paths)):
        paths[i] = fix_path(paths[i])
    return " ".join(paths)


df = pd.read_csv(NODES_VOCABULARY)
node_to_id = {k: v for v, k in zip(df["id"].tolist(), df["nodeType"].tolist())}
dataset_path = "separate_lines"
for projects_set in ["training", "validation", "test"]:
    projects_set_folder = os.path.join(dataset_path, projects_set)
    for name in os.listdir(projects_set_folder):
        input_data = os.path.join(projects_set_folder, name, "result.c2s")
        nodes_data = os.path.join(projects_set_folder, name, "nodes_vocabulary.csv")
        df1 = pd.read_csv(nodes_data)
        id_to_node = {k: v for k, v in zip(df1["id"].tolist(), df1["nodeType"].tolist())}
        correct_lines = []
        with open(input_data, "r") as f:
            for line in f:
                correct_lines.append(process_sample(line))
        with open(input_data, "w") as f:
            f.writelines(correct_lines)
