import gtnlplib.constants
from collections import defaultdict
from gtnlplib.constants import END_TAG, START_TAG, TRANS, CURR_SUFFIX, PREV_SUFFIX

def wordFeatures(words,tag,prev_tag,m):
    '''
    :param words: a list of words
    :type words: list
    :param tag: a tag
    :type tag: string
    :type prev_tag: string
    :type m: int
    '''
    out = {(gtnlplib.constants.OFFSET,tag):1}
    if m < len(words): #we can have m = M, for the transition to the end state
        out[(gtnlplib.constants.EMIT,tag,words[m])]=1
    return out

def wordCharFeatures(words,tag,prev_tag,m):
    output = wordFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    # add your code here
    if m < len(words):
        output[(CURR_SUFFIX, tag, words[m][-1])] = 1

    if m > 0:
        output[(PREV_SUFFIX, tag, words[m-1][-1])] = 1

    return output

def yourFeatures(words,tag,prev_tag,m):
    output = wordCharFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    # your stuff
    return output

def seqFeatures(words,tags,featfunc):
    '''
    :param words: a list of words
    :param tags: a list of tags
    :param featfunc: a function to compute f(words,tag_m,tag_{m-1},m)
    :returns list of features
    '''
    allfeats = defaultdict(float)
    # your code here
    for m in xrange(len(words) + 1):
        prev_tag = tags[m-1] if m > 0 else START_TAG
        tag = END_TAG if m == len(words) else tags[m]
        for feat in featfunc(words, tag, prev_tag, m):
            allfeats[feat] += 1

    return allfeats


def wordTransFeatures(words,tag,prev_tag,m):
    output = wordFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    # your code here
    output[(TRANS, tag, prev_tag)] = output.get((TRANS, tag, prev_tag),0) + 1
    return output

def yourHMMFeatures(words,tag,prev_tag,m):
    output = wordTransFeatures(words,tag,prev_tag,m) #start with the features from wordFeatures
    #add smart stuff
    return output