import os
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
from ioffe_plda.verifier import Verifier
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

lda_si_embeds = np.load("xvector_embeds/voxc1_si_lda_embeds.npy")
si_labels = np.load("xvector_embeds/voxc1_si_labels.npy")
lda_sv_embeds = np.load("xvector_embeds/voxc1_sv_lda_embeds.npy")
sv_labels = np.load("xvector_embeds/voxc1_sv_labels.npy")

ln_lda_si_embeds = lda_si_embeds * np.sqrt(lda_si_embeds.shape[1]) / \
                        np.linalg.norm(lda_si_embeds, axis=1, keepdims=True)

ln_lda_sv_embeds = lda_sv_embeds * np.sqrt(lda_sv_embeds.shape[1]) / \
                        np.linalg.norm(lda_sv_embeds, axis=1, keepdims=True)

plda_model_file = "xvector_embeds/plda_model.pkl"
if not os.path.isfile(plda_model_file):
    plda_verifier = Verifier()
    plda_verifier.fit(ln_lda_si_embeds, si_labels)
    pickle.dump(plda_verifier, open(plda_model_file, 'wb'))
else:
    plda_verifier = Verifier(plda_model_file)

trial = pd.read_csv("voxc1_sv_trial.csv")

score_vector = []
for i, row in tqdm(trial.iterrows(), total=len(trial), dynamic_ncols=True):
    score= plda_verifier.multi_sess(ln_lda_sv_embeds[row.enroll_idx],
            ln_lda_sv_embeds[row.test_idx], cov_scaling=True).squeeze()
    score_vector.append(score)

label_vector = np.array(trial.label)

fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

print("EER: {:.4f}".format(eer))

