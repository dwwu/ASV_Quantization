import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import argparse

def cosine_similarity(a, b):
    """
    a: shape=(n_samples, dim)
    b: shape=(n_samples, dim)
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot_product = (a * b).sum(1)

    return dot_product

parser = argparse.ArgumentParser("sv test")
parser.add_argument("-embed_file", type=str, required=True)
args = parser.parse_args()

trial = pd.read_csv("voxc1_sv_trial.csv")
embeddings = np.load(args.embed_file).squeeze()

score_vector = cosine_similarity(embeddings[trial.enroll_idx],
        embeddings[trial.test_idx])
label_vector = np.array(trial.label)

fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

print("EER: {:.4f}".format(eer))

