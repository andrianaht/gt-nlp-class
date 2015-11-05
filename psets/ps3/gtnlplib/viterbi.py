import numpy as np #hint: np.log
from itertools import chain
import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conllSeqGenerator

from gtnlplib import scorer
from gtnlplib import most_common
from gtnlplib import preproc
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT
from gtnlplib import naivebayes

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

# define viterbiTagger
def viterbiTagger(words,feat_func,weights,all_tags,debug=False):
    """Tag the given words using the viterbi algorithm
        Parameters:
        words -- A list of tokens to tag
        feat_func -- A function of (words, curr_tag, prev_tag, curr_index)
        that produces features
        weights -- A defaultdict that maps features to numeric score. Should
        not key error for indexing into keys that do not exist.
        all_tags -- A set of all possible tags
        debug -- (optional) If True, print the trellis at each layer
        Returns:
        tuple of (tags, best_score), where
        tags -- The highest scoring sequence of tags (list of tags s.t. tags[i]
        is the tag of words[i])
        best_score -- The highest score of any sequence of tags
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
                # prev_tag = START_TAG
                # emission, transmission = feat_func(words, tag, prev_tag, k)
                trellis[k][tag] = sum([weights[feat] for feat in feat_func(words, tag, START_TAG, k)])

                # if debug:
                #     print "{} := q({}/{}) + e({}/{}) = {} {} = {}".\
                #                 format(k+1, tag, prev_tag, word, tag, weights[transmission],weights[emission], trellis[k][tag])

            else:
                for prev_tag in all_tags:
                    # emission, transmission = feat_func(words, tag, prev_tag, k)
                    temp[prev_tag] = sum([weights[feat] for feat in feat_func(words, tag, prev_tag, k)])+trellis[k-1][prev_tag]

                # if debug:
                #     for prev_tag, val in temp.iteritems():
                #         print "{} := q({}/{}) + e({}/{}) = {} {} {} = {}".\
                #             format(k+1, tag, prev_tag, word, tag, weights[transmission],weights[emission], trellis[k-1][prev_tag], val)

                trellis[k][tag] = max(temp.values())
                pointers[k][tag] = argmax(temp)
            # output[k] = argmax(trellis[k])

        if debug:
            print trellis[k]
            print pointers[k]

    for tag in all_tags:
        score = trellis[-1][tag] + weights[(END_TAG, tag, TRANS)]
        if score > best_score:
            output[-1] = tag
            best_score = score

    for i in range(len(words)-1, 0, -1):
        output[i-1] = pointers[i][output[i]]
    return output, best_score

def get_HMM_weights(trainfile):
    """Train a set of of log-prob weights using HMM transition model
        Parameters:
        trainfile -- The name of the file to train weights
        Returns:
        weights -- Weights dict with log-prob of transition and emit features

        ngrams("I really like", 2)
            (I, really)
            (really, like)
        (end_tag,'N',trans)  => q(stop/N) => (N, end_tag)
        q(stop/N) = count(stop, N) / count(N)
        """
    # compute naive bayes weights
    counters = most_common.get_tags(trainfile)
    class_counts = most_common.get_class_counts(counters)
    allwords = set()
    for counts in counters.values():
        allwords.update(set(counts.keys()))

    nb_weights = naivebayes.learnNBWeights(counters, class_counts, allwords, alpha=0.001)

    # convert nb weights to hmm weights
    hmm_weights = defaultdict(lambda: -1000.)
    for (tag, word), weight in nb_weights.iteritems():
        hmm_weights[(tag, word, EMIT)] = weight

    unigram = preproc.getNgrams(trainfile)
    bigram = preproc.getNgrams(trainfile, 2)
    unigramCount = preproc.getAllCounts(unigram)
    bigramCount = preproc.getAllCounts(bigram)

    for (tag1, tag2) in bigramCount.keys():
        hmm_weights[(tag2, tag1, TRANS)] = np.log(1.*bigramCount.get((tag1, tag2), 0)) - np.log(unigramCount.get(tag1, 0))

    return hmm_weights

def hmm_feats(words,curr_tag,prev_tag,i):
    """Feature function for HMM that returns emit and transition features"""
    if i < len(words):
        return [(curr_tag,words[i],EMIT),(curr_tag,prev_tag,TRANS)]
    else:
        return [(curr_tag,prev_tag,TRANS)]