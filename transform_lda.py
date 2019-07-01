import os
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_curve

parser = argparse.ArgumentParser("Transform embed through LDA")
parser.add_argument("-dev_file", type=str, required=True, help="dev embed file")
parser.add_argument("-test_file", type=str, required=True, help="test embed file")
parser.add_argument("-lda_file", type=str, required=True, help="to be saved lda file")
parser.add_argument("-lda_dim", type=int, required=False, default=200, help="LDA output dimensions")
args = parser.parse_args()

def cosine_similarity(a, b):
    """
    a: shape=(n_samples, dim)
    b: shape=(n_samples, dim)
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot_product = (a * b).sum(1)

    return dot_product

def suffix_file_name(filen, suffix):
    return filen.rstrip(".npy") + "_" + suffix + ".npy"


#####################################################
# parameters
#####################################################

dev_file = args.dev_file
test_file = args.test_file
lda_file = lda.lda_file


#####################################################
# load embeds
#####################################################

dev_embeds = np.load(dev_file)
dev_labels = np.load(suffix_file_name(dev_file, "label"))
test_embeds = np.load("xvector_embeds/voxc1_sv_embeds.npy")

#####################################################
# LDA model training and trasform embeds
#####################################################

dev_mean = dev_embeds.mean(0)
centered_dev_embeds = dev_embeds - dev_mean.reshape(1, -1)
centered_test_embeds = test_embeds- dev_mean.reshape(1, -1)

if not os.path.isfile(lda_file):
    clf = LDA(solver='svd', n_components=200)
    clf.fit(centered_si_embeds, dev_labels)
    pickle.dump(clf, open(lda_file, "wb"))
else:
    clf = pickle.load(open(lda_file, "rb"))

embed_dir = os.path.dirname(dev_file)
lda_dev_embeds = clf.transform(centered_dev_embeds)
np.save(suffix_file_name(dev_file,, 'lda'), lda_dev_embeds)
lda_test_embeds = clf.transform(centered_test_embeds)
np.save(suffix_file_name(test_file,, 'lda'), lda_test_embeds)

#####################################################
# ASV scoring with LDA transformed embeds 
#####################################################

trial = pd.read_csv("voxc1_sv_trial.csv")
score_vector = cosine_similarity(lda_test_embeds[trial.enroll_idx],
        lda_test_embeds[trial.test_idx])
label_vector = np.array(trial.label)

fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

print("[LDA] EER: {:.4f}".format(eer))

#####################################################
# PLDA model training and transform embeds
#####################################################
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
