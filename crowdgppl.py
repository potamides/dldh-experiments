import pickle
from sys import path
from glob import glob
from os import getcwd
from os.path import join, realpath, dirname, exists
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
__location__ = realpath(join(getcwd(), dirname(__file__)))
path.append(join(__location__, "crowdGPPL", "python", "models"))
from crowdGPPL.python.models.collab_pref_learning_svi import CollabPrefLearningSVI

EMBEDDINGS_FILE = join("embeddings", "embeddings.pkl")
MODEL_FILE = join("models", "model.pkl")
POEM_FOLDER = "poems"


def embed_sentences(sentences):
    if exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            saved_sents, embeddings = pickle.load(f)
            if saved_sents == sentences:
                return embeddings

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump((sentences, embeddings), f)

    return embeddings


def load_train_dataset():
    person_train = []
    a1_sent_idx = []
    a2_sent_idx = []
    prefs_train = []
    sents = []

    for filename in glob(join(POEM_FOLDER, "*.tsv")):

        with open(filename, 'r') as f:
            categories = f.readline().strip("\n").split("\t")[2:]

            for line in f.readlines():
                line = line.split("\t")
                for text in line[0:2]:
                    if text not in sents:
                        sents.append(text)
                for idx, category in enumerate(categories, 2):
                    label = line[idx]

                    if label.strip() == "1":
                        prefs_train.append(-1)
                    elif label.strip() == "2":
                        prefs_train.append(1)
                    elif label.strip():
                        prefs_train.append(0)
                    else:
                        continue

                    person_train.append(category)
                    a1_sent_idx.append(sents.index(line[0]))
                    a2_sent_idx.append(sents.index(line[1]))

    sent_features = embed_sentences(sents)

    ndims = len(sent_features[0])

    id2idx = dict([(v, k) for k, v in dict(
        enumerate(np.unique(person_train))).items()])

    person_train_idx = np.array([id2idx[id_] for id_ in person_train], dtype=int)
    a1_sent_idx = np.array(a1_sent_idx, dtype=int)
    a2_sent_idx = np.array(a2_sent_idx, dtype=int)
    prefs_train = np.array(prefs_train, dtype=float)

    return person_train_idx, a1_sent_idx, a2_sent_idx, sent_features, prefs_train, ndims


def train_model():
    # Train a model...
    person_train_idx, a1_train, a2_train, sent_features, prefs_train, ndims = load_train_dataset()

    model = CollabPrefLearningSVI(
        ndims, shape_s0=2, rate_s0=200, use_lb=True, use_common_mean_t=True, ls=None)

    model.max_iter = 600

    model.fit(person_train_idx, a1_train, a2_train, sent_features, prefs_train, optimize=False,
              use_median_ls=True, input_type='zero-centered')

    logging.info("**** Completed training GPPL ****")

    # Save the model in case we need to reload it

    with open(MODEL_FILE, 'wb') as fh:
        pickle.dump(model, fh)


def get_crowd_score(sentences):
    with open(MODEL_FILE, 'rb') as fh:
        model = pickle.load(fh)
    sentences = [sent.replace("\n", " ").replace("_", "")
                 for sent in sentences.split("\n\n")]
    sentences = embed_sentences(sentences)
    #print(model.predict_t().size)
    return np.asarray(model.predict_t(sentences)).mean()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
