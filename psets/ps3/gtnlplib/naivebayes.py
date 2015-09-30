import numpy as np #hint: np.log
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

# this is from pset 1
def learnNBWeights(counts, class_counts, allkeys, alpha=0.1):
    weights = defaultdict(int)
    vocabulary_len = 1.*len(allkeys)
    corpus_len = 1.*sum(class_counts.values())
    for label in counts.keys():
        prior = class_counts[label]/corpus_len
        normilizer = 1.*sum(counts[label].values())

        for word in allkeys:
            pwy = (counts[label].get(word, 0) + alpha) / (normilizer + alpha * vocabulary_len)
            weights.update({(label, word): np.log(pwy)})
        weights.update({(label, OFFSET): np.log(prior)})
    return weights