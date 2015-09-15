import numpy as np #hint: np.log
from itertools import chain
from collections import defaultdict, Counter
from gtnlplib.preproc import dataIterator
from gtnlplib.constants import OFFSET, TRAINKEY, DEVKEY
from gtnlplib import scorer
from gtnlplib.clf_base import evalClassifier

''' keep the shell '''
# weights_all_pos.update({('POS',OFFSET):1,('NEG',OFFSET):0,('NEU',OFFSET):0})
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

def regularization_using_grid_search(alphas, counts, class_counts, allkeys, tr_outfile='nb.alpha.tr.txt', dv_outfile='nb.alpha.dv.txt'):
    tr_accs = []
    dv_accs = []
    # Choose your alphas here
    weights_nb_alphas = dict()
    for alpha in alphas:
        weights_nb_alphas[alpha] = learnNBWeights(counts, class_counts, allkeys, alpha)
        confusion = evalClassifier(weights_nb_alphas[alpha],tr_outfile,TRAINKEY)
        tr_accs.append(scorer.accuracy(confusion))
        confusion = evalClassifier(weights_nb_alphas[alpha],dv_outfile,DEVKEY)
        dv_accs.append(scorer.accuracy(confusion))
    return weights_nb_alphas, tr_accs, dv_accs
