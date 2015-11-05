import gtnlplib.constants
from collections import defaultdict
from gtnlplib.constants import END_TAG, START_TAG, TRANS, CURR_SUFFIX, PREV_SUFFIX

from gtnlplib.constants import CURR_PREFFIX, ADJ_TAG
import re

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

    # Suffix Tag
    output[(CURR_SUFFIX, tag, words[m][-4:])] = 1   # -able
    output[(CURR_SUFFIX, tag, words[m][-2:])] = 1   # -ly
    output[(CURR_SUFFIX, tag, words[m][-3:])] = 1   # -ing
    # Prefix Tag
    output[(CURR_PREFFIX, tag, words[m][:2])] = 1   # un-
    output[(CURR_PREFFIX, tag, words[m][:1])] = 1   # #, @

    output[('--word-tag--', words[m], tag)] = 1    # <W: shitty, JJ>
    output[(ADJ_TAG, tag, prev_tag)] = 1    # DT N | O V | JJ MNS
    output[('--all-cap--', words[m].upper() == words[m], tag)] = 1
    if m < len(words)-1:
        output[('--next-word--', words[m+1], tag)] = 1   # <N: shitty, DT>

    if m > 0:
        output[('--prev-word--', words[m-1], tag)] = 1   # <N: shitty, DT>
        output[('--first-cap--'), words[m][0].upper() == words[m][0], tag] = 1
        # output[('--first-cap--'), tag, prev_tag] = words[m][0].upper() == words[m][0]


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
    if m < len(words):
        # Suffixes Tag
        output[(CURR_SUFFIX), tag, words[m][-4:]] = 1   # -able
        output[(CURR_SUFFIX), tag, words[m][-3:]] = 1   # -ing
        output[(CURR_SUFFIX), tag, words[m][-2:]] = 1   # -ly
        # output[(CURR_SUFFIX), tag, words[m][-1:]] = 1   # -y
        # Prefix Tag
        output[(CURR_PREFFIX, tag, words[m][:2])] = 1   # un-
        output[(CURR_PREFFIX, tag, words[m][:1])] = 1   # #, @

        # if m > 0:
        output[('--all-cap--', tag)] = 1
        output[('--first-cap--'), tag] = 1

        output[('--length--'), len(words[m])] = 1

        if m > 0:
            output[('--length-1--'), len(words[m-1])] = 1

        if m < len(words) - 1:
            output[('--length-2--'), len(words[m+1])] = 1

    return output


"""
Features:

Base  + all-cap => 0.8496
Base + all-cap + prev-word + first-cap => 0.8486

Base: Suffixes Tag + Prefix Tag => 83.18
    1) ( ^, N ): 68 and (N, ^): 33
    2) (N, V): 44 and (V, N): 42
    3) (A, N): 36

Base + First Cap Feat => 83.53

A N 42
V N 37
N ^ 28
A V 21
^ N 64
^ V 22
N V 47



A N 40
V N 45
N ^ 33
^ N 66
A V 22
^ V 28
N V 42


 # bigram
        # output[('--bigram--', (words[m], words[m+1] if m < len(words)-1 else None), tag)] = 1
        # if re.match("[\W]", words[m]) and len(set(words[m])) < 4:
        #     output[('--punct--'), ''.join(set(words[m])), tag] = 1

        # (^, N): 68
        # output[('--first-cap--'), tag, prev_tag] = words[m][0].upper() == words[m][0] 0.846568525814

        # if(words[m][0].upper() == words[m][0]):
        #     output[('--first-cap--'), words[m], tag] = 1

        # if m > 0:
        #     output[('--prev-word--', words[m-1], tag)] = 1   # <N: shitty, DT>

        # if m < len(words)-1:
        #     output[('--next-word--', words[m+1], tag)] = 1   # <A: broken, N>

        # output[('--prev-prev-word--', words[m-2] if m > 1 else None, tag)] = 1
        # output[('--next-next-word--', words[m+2] if m < len(words)-2 else None, tag)] = 1
        # output[('--next-word--', words[m+1] if m < len(words)-1 else None, tag)] = 1
        # output[('--word-tag--', words[m], tag)] = 1    # <W: shitty, JJ>
        # output[(ADJ_TAG, tag, prev_tag)] = 1    # DT N | O V | JJ MNS

"""