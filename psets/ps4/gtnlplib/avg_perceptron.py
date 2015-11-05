from collections import defaultdict
from gtnlplib.tagger_base import classifierTagger
from gtnlplib.tagger_base import evalTagger 
from gtnlplib import scorer
from gtnlplib.constants import START_TAG, END_TAG

def oneItAvgPerceptron(inst_generator,featfunc,weights,wsum,tagset,Tinit=0):
    """
    :param inst_generator: iterator over instances
    :param featfunc: feature function on (words, tag_m, tag_m_1, m)
    :param weights: default dict
    :param wsum: weight sum, for averaging
    :param tagset: set of permissible tags
    :param Tinit: initial value of t, the counter over instances
    """
    tr_err = 0.0
    for i,(words,y_true) in enumerate(inst_generator):
        # your code here
        y_pred = classifierTagger(words, featfunc, weights, tagset)
        for m in xrange(len(words)):
            if y_pred[m] != y_true[m]:
                prev_tag_pred = y_pred[m-1] if m > 0 else START_TAG
                prev_tag_true = y_true[m-1] if m > 0 else START_TAG

                for feat, value in featfunc(words, y_true[m], prev_tag_true, m).iteritems():
                    wsum[feat] += (Tinit+i)*value
                    weights[feat] += value

                for feat, value in featfunc(words, y_pred[m], prev_tag_pred, m).iteritems():
                    wsum[feat] -= (Tinit+i)*value
                    weights[feat] -= value
                tr_err += 1

    # note that i'm computing tr_acc for you, as long as you properly update tr_err
    return weights, wsum, 1.-tr_err / float(sum([len(s) for s,t in inst_generator])), i


def trainAvgPerceptron(N_its,inst_generator,featfunc,tagset):
    """
    :param N_its: number of iterations
    :param inst_generator: generate words,tags pairs
    :param featfunc: feature function
    :param tagset: set of all possible tags
    :returns average weights, training accuracy, dev accuracy
    """
    tr_acc = [None]*N_its
    dv_acc = [None]*N_its
    T = 0
    weights = defaultdict(float)
    wsum = defaultdict(float)
    avg_weights = defaultdict(float)

    for i in xrange(N_its):
        # your code here
        weights, wsum, tr_acc_i, tot = oneItAvgPerceptron(inst_generator,featfunc,weights,wsum,tagset,T)
        T += tot
        for key in weights:
            avg_weights[key] = weights[key] - wsum[key]/T

        confusion = evalTagger(lambda words, alltags: classifierTagger(words,featfunc,avg_weights,tagset),'perc')
        dv_acc[i] = scorer.accuracy(confusion)
        tr_acc[i] = tr_acc_i
        print i,'dev:',dv_acc[i],'train:',tr_acc[i]
    return avg_weights, tr_acc, dv_acc