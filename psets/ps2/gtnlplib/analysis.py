import operator
import numpy as np
from collections import defaultdict

# {('POS',OFFSET):1,('NEG',OFFSET):0,('NEU',OFFSET):0})
def getTopFeats(weights,class1,class2,allkeys,K=5):
    logratio = defaultdict(int)
    for word in allkeys:
        logratio[word] = weights[(class1, word)] - weights[(class2, word)]

    return sorted(logratio.iteritems(), key=operator.itemgetter(1), reverse=True)[:K] # reverse=True