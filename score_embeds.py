import os
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_curve
from ioffe_plda.verifier import Verifier

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transform embed through LDA")
    parser.add_argument("-dev_file", type=str, required=True, help="dev embed file")
    parser.add_argument("-test_file", type=str, required=True, help="test embed file")
    parser.add_argument("-lda_dim", type=int, required=False, default=200, help="LDA output dimensions")
    args = parser.parse_args()

    #####################################################
    # parameters
    #####################################################

    dev_file = args.dev_file
    test_file = args.test_file
    embed_dir = os.path.dirname(dev_file)
    lda_file = os.path.join(embed_dir, "lda_model.pkl")
    plda_file = os.path.join(embed_dir, "plda_model.pkl")

    #####################################################
    # load embeds
    #####################################################

    dev_embeds = np.load(dev_file).squeeze()
    dev_labels = np.load(os.path.join(embed_dir, 'dev_label.npy'))
    test_embeds = np.load(test_file).squeeze()

    #####################################################
    # LDA model training and trasform embeds
    #####################################################

    dev_mean = dev_embeds.mean(0)
    centered_dev_embeds = dev_embeds - dev_mean.reshape(1, -1)
    centered_test_embeds = test_embeds - dev_mean.reshape(1, -1)

    if not os.path.isfile(lda_file):
        clf = LDA(solver='svd', n_components=200)
        clf.fit(centered_dev_embeds, dev_labels)
        pickle.dump(clf, open(lda_file, "wb"))
    else:
        clf = pickle.load(open(lda_file, "rb"))

    lda_dev_embeds = clf.transform(centered_dev_embeds)
    np.save(suffix_file_name(dev_file, 'lda'), lda_dev_embeds)
    lda_test_embeds = clf.transform(centered_test_embeds)
    np.save(suffix_file_name(test_file, 'lda'), lda_test_embeds)

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

    ln_lda_dev_embeds = lda_dev_embeds * np.sqrt(lda_dev_embeds.shape[1]) / \
                            np.linalg.norm(lda_dev_embeds, axis=1, keepdims=True)

    ln_lda_test_embeds = lda_test_embeds * np.sqrt(lda_test_embeds.shape[1]) / \
                            np.linalg.norm(lda_test_embeds, axis=1, keepdims=True)

    if not os.path.isfile(plda_file):
        plda_verifier = Verifier()
        plda_verifier.fit(ln_lda_dev_embeds, dev_labels)
        pickle.dump(plda_verifier, open(plda_file, 'wb'))
    else:
        plda_verifier = Verifier(plda_file)

    #####################################################
    # ASV scoring with PLDA transformed embeds
    #####################################################

    trial = pd.read_csv("voxc1_sv_trial.csv")

    score_vector = []
    for i, row in tqdm(trial.iterrows(), total=len(trial), dynamic_ncols=True):
        score= plda_verifier.multi_sess(ln_lda_test_embeds[row.enroll_idx],
                ln_lda_test_embeds[row.test_idx], cov_scaling=True).squeeze()
        score_vector.append(score)

    label_vector = np.array(trial.label)

    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    print("[PLDA] EER: {:.4f}".format(eer))

