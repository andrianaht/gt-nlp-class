import numpy as np #hint: np.log
from itertools import chain
import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conllSeqGenerator

from gtnlplib import scorer
from gtnlplib import constants
from gtnlplib import preproc
from gtnlplib.constants import START_TAG ,TRANS ,END_TAG , EMIT

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

# define viterbiTagger
start_tag = constants.START_TAG
trans = constants.TRANS
end_tag = constants.END_TAG
emit = constants.EMIT


def viterbiTagger(words,feat_func,weights,all_tags,debug=False):
    """
    :param words: list of words
    :param feat_func: feature function
    :param weights: defaultdict of weights
    :param tagset: list of permissible tags
    :param debug: optional debug flag
    :returns output: tag sequence
    :returns best_score: viterbi score of best tag sequence
    """

    trellis = [None] * len(words)
    pointers = [None] * len(words)
    output = [None] * len(words)
    best_score = -np.inf

    for k, word in enumerate(words):
        trellis[k] = defaultdict(lambda: -1000.)
        pointers[k] = defaultdict(str)
        for tag in all_tags:
            temp = defaultdict(lambda: -1000.)
            if k == 0:
                trellis[k][tag] = sum([weights[feat]*value for feat, value in feat_func(words, tag, START_TAG, k).iteritems()])
            else:
                for prev_tag in all_tags:
                    temp[prev_tag] = sum([weights[feat]*value for feat, value in feat_func(words, tag, prev_tag, k).iteritems()])+trellis[k-1][prev_tag]

                trellis[k][tag] = max(temp.values())
                pointers[k][tag] = argmax(temp)

        # if debug:
        #     print 'trellis  :', trellis[k]
        #     print 'pointers :', pointers[k]
        #     print
        #     print

    for tag in all_tags:
        score = trellis[-1][tag] + sum([weights[feat]*value for feat, value in feat_func(words, END_TAG, tag, k).iteritems()])
        if score > best_score:
            output[-1] = tag
            best_score = score

    for i in range(len(words)-1, 0, -1):
        output[i-1] = pointers[i][output[i]]
    return output, best_score


