import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def cosine_similarity(a, b):
    """
    a: shape=(n_samples, dim)
    b: shape=(n_samples, dim)
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot_product = (a * b).sum(1)

    return dot_product

trial = pd.read_csv("voxc1_sv_trial.csv")
embeddings = np.load("xvector_embeds/voxc1_sv_embeds.npy")

score_vector = cosine_similarity(embeddings[trial.enroll_idx],
        embeddings[trial.test_idx])
label_vector = np.array(trial.label)

fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

print("EER: {:.4f}".format(eer))

