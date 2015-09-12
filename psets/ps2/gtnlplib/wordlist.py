from collections import defaultdict
from gtnlplib.constants import OFFSET

def loadSentimentWords (sentiment_file):
    """
    Loads the sentiment words from a file.
    Returns a two things: set of positive words and set of negative words
    """
    poswords = set()
    negwords = set()

    with open(sentiment_file,'r') as fin:
        for i,line in enumerate(fin):
            # more list and dict comprehensions!
            kvs = {key:val for key,val in [kvp.split('=') for kvp in line.split() if '=' in kvp]}
            if kvs['type'] == 'strongsubj':
                if kvs['priorpolarity'] == 'negative':
                    negwords.add(kvs['word1'])
                if kvs['priorpolarity'] == 'positive':
                    poswords.add(kvs['word1'])

    return poswords, negwords


def learnMCCWeights():
    weights_all_pos = defaultdict(int)
    weights_all_pos.update({('POS',OFFSET):1,('NEG',OFFSET):0,('NEU',OFFSET):0})
    return weights_all_pos

def learnWLCWeights (poswords, negwords):
    weights_list = defaultdict(int)

    for word in poswords:
        weights_list.update({('POS', word): 1})

    for word in negwords:
        weights_list.update({('NEG', word): 1})
    return weights_list