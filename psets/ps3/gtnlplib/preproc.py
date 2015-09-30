from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import defaultdict, Counter
import os.path
from itertools import chain
from gtnlplib.constants import OFFSET, UNKNOWN
from gtnlplib.constants import END_TAG, START_TAG

def getAllTags(input_file):
    """Return unique set of tags in the conll file"""
    alltags = set([])
    for _, tags in conllSeqGenerator(input_file):
        for tag in tags:
            alltags.add(tag)
    return alltags

def conllSeqGenerator(input_file,max_insts=1000000):
    """Create a generator of (words, tags) pairs over the conll input file
    
    Parameters:
    input_file -- The name of the input file
    max_insts -- (optional) The maximum number of instances (words, tags)
                 instances to load
    returns -- generator of (words, tags) pairs
    """
    cur_words = []
    cur_tags = []
    num_insts = 0
    with open(input_file) as instances:
        for line in instances:
            if num_insts >= max_insts:
                return

            if len(line.rstrip()) == 0:
                if len(cur_words) > 0:
                    num_insts += 1
                    yield cur_words,cur_tags
                    cur_words = []
                    cur_tags = []
            else:
                parts = line.rstrip().split()
                cur_words.append(parts[0])
                if len(parts)>1:
                    cur_tags.append(parts[1])
                else: cur_tags.append(UNKNOWN)
        if num_insts >= max_insts:
           return

        if len(cur_words)>0:
            num_insts += 1
            yield cur_words,cur_tags


def getAllCounts(datait):
    allcounts = defaultdict(int)
    for gram in datait:
        allcounts[gram] += 1
    return allcounts

def getNgrams(input_file, window_size=1):
    all_instances = []
    # all_instances = [tag for (words, tags) in conllSeqGenerator(input_file) for tag in tags]
    for (words, tags) in conllSeqGenerator(input_file):
        all_instances.append(START_TAG)
        for tag in tags:
            all_instances.append(tag)
        all_instances.append(END_TAG)
    return all_instances if window_size == 1 else ngrams(all_instances, window_size)