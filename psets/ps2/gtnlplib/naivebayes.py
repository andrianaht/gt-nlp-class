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
    # v = 1.*sum([count for label in counts.keys() for count in counts[label].values()])
    v = 1.*len(allkeys)-1
    N = 1.*sum(class_counts.values())

    # log(p(y)) + log(p(w/y))you're lovely beside live in the world something is happening
    for label in counts.keys():
        py = class_counts[label]/N
        z = 1.*sum(counts[label].values())

        for word in allkeys - set(OFFSET):
            pwy = (counts[label].get(word, 0) + alpha) / (z + alpha*v)
            weights.update({(label, word): np.log(pwy)})

        weights.update({(label, OFFSET): np.log(py)})

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
