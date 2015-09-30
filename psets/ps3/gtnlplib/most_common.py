''' your code '''
import operator
from  constants import *
from collections import defaultdict, Counter
from gtnlplib import preproc
import scorer
from gtnlplib import constants
from gtnlplib import clf_base

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tags(trainfile):
    """Produce a Counter of occurences of word in each tag"""
    counters = Counter()
    for (words, tags) in preproc.conllSeqGenerator(trainfile):
        for i, tag in enumerate(tags):
            if counters[tag] == 0:
                counters[tag] = Counter()
            counters[tag][words[i]] += 1
    return counters

def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    your_weights = defaultdict(lambda: 0)
    your_weights.update({('N', OFFSET): 1})
    return your_weights

def get_most_common_weights(trainfile):
    weights = defaultdict(int)
    alltags = preproc.getAllTags(trainfile)
    tag_ctrs = get_tags(constants.TRAIN_FILE)
    allwords = set()
    for (words, tags) in preproc.conllSeqGenerator(trainfile):
            allwords.update(set(words))
    for word in allwords:
        max_weight = 0
        max_tag = None
        for tag in alltags:
            weights.update({(tag, word): 0})
            weight = tag_ctrs.get(tag, {}).get(word, 0)
            if weight > max_weight:
                max_tag = tag
                max_weight = tag_ctrs.get(tag, {}).get(word, 0)
        weights.update({(max_tag, word): max_weight})
    return weights

def get_class_counts(counters):
    return {tag : sum(tag_ctr.values()) for tag, tag_ctr in counters.iteritems()}