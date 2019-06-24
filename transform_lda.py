import os
import pandas as pd
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

si_embeds = np.load("xvector_embeds/voxc1_si_embeds.npy")
si_labels = np.load("xvector_embeds/voxc1_si_labels.npy")

sv_embeds = np.load("xvector_embeds/voxc1_sv_embeds.npy")
sv_labels = np.load("xvector_embeds/voxc1_sv_labels.npy")

si_embed_mean = si_embeds.mean(0)
centered_si_embeds = si_embeds - si_embed_mean.reshape(1, -1)
centered_sv_embeds = sv_embeds- si_embed_mean.reshape(1, -1)

lda_model_file = "xvector_embeds/lda_model.pkl"
if not os.path.isfile(lda_model_file):
    clf = LDA(solver='svd', n_components=200)
    clf.fit(centered_si_embeds, si_labels)
    pickle.dump(clf, open("lda_model_file", "wb"))
else:
    clf = pickle.load(open(lda_model_file, "rb"))

lda_si_embeds = clf.transform(centered_si_embeds)
np.save("xvector_embeds/voxc1_si_lda_embeds.npy", lda_si_embeds)
lda_sv_embeds = clf.transform(centered_sv_embeds)
np.save("xvector_embeds/voxc1_sv_lda_embeds.npy", lda_sv_embeds)

trial = pd.read_csv("voxc1_sv_trial.csv")
score_vector = cosine_similarity(lda_sv_embeds[trial.enroll_idx],
        lda_sv_embeds[trial.test_idx])
label_vector = np.array(trial.label)

fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

print("EER: {:.4f}".format(eer))


