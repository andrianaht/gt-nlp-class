from collections import defaultdict, Counter
from gtnlplib.tagger_base import classifierTagger
from gtnlplib.tagger_base import evalTagger 
from gtnlplib import scorer
from gtnlplib.viterbi import viterbiTagger
from gtnlplib.features import seqFeatures
from gtnlplib.constants import START_TAG, TRANS

def oneItAvgStructPerceptron(inst_generator,
                             featfunc,
                             weights,
                             wsum,
                             tagset,
                             Tinit=0):
    """
    :param inst_generator: A generator of (words,tags) tuples
    :param tagger: A function from (words, weights) to tags
    :param features: A function from (words, tags) to a dict of features and weights
    :param weights: A defaultdict of weights
    :param wsum: A defaultdict of weight sums
    :param Tinit: the initial value of the $t$ counter at the beginning of this iteration
    :returns weights: a defaultdict of weights
    :returns wsum: a defaultdict of weight sums, for averaging
    :returns tr_acc: the training accuracy
    :returns i: the number of instances (sentences) seen
    """
    tr_err = 0.
    tr_tot = 0.
    # your code
    for i,(words,y_true) in enumerate(inst_generator):
        y_pred, score = viterbiTagger(words, featfunc, weights, tagset)


        # if '!' not in y_true and i  > 0:
        #     print i, words
        #     print y_true
        #     # Make sure features is right
        #     for feat, value in seqFeatures(words, y_true, featfunc).iteritems():
        #         print feat, value
        #
        #     print



        if y_pred != y_true:
            for feat, value in seqFeatures(words, y_true, featfunc).iteritems():
                wsum[feat] += (Tinit+i)*value
                weights[feat] += value
            for feat, value in seqFeatures(words, y_pred, featfunc).iteritems():
                wsum[feat] -= (Tinit+i)*value
                weights[feat] -= value
            tr_err += sum([y_true[m] != y_pred[m] for m, _ in enumerate(y_true)])
        tr_tot += len(words)
    return weights, wsum, 1-tr_err/tr_tot, i

def trainAvgStructPerceptron(N_its,inst_generator,featfunc,tagset):
    """
    :param N_its: number of iterations
    :param inst_generator: A generator of (words,tags) tuples
    :param tagger: A function from (words, weights) to tags
    :param features: A function from (words, tags) to a dict of features and weights
    """

    tr_acc = [None]*N_its
    dv_acc = [None]*N_its
    T = 0
    weights = defaultdict(float)
    wsum = defaultdict(float)
    avg_weights = defaultdict(float)
    for i in xrange(N_its):
        # your code here
        # note that I call evalTagger to produce the dev set results
        weights, wsum, tr_acc_i, tot = oneItAvgStructPerceptron(inst_generator,featfunc,weights,wsum,tagset,T)
        T += tot
        for key in weights:
            avg_weights[key] = weights[key] - wsum[key]/T

        confusion = evalTagger(lambda words,tags : viterbiTagger(words,featfunc,avg_weights,tags)[0],'sp.txt')
        dv_acc[i] = scorer.accuracy(confusion)
        tr_acc[i] = tr_acc_i#1. - tr_err/float(sum([len(s) for s,t in inst_generator]))
        print i,'dev:',dv_acc[i],'train:',tr_acc[i]
    return avg_weights, tr_acc, dv_acc