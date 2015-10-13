from collections import defaultdict
from gtnlplib.tagger_base import classifierTagger
from gtnlplib.tagger_base import evalTagger 
from gtnlplib import scorer
from gtnlplib.constants import START_TAG

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
        if y_pred != y_true:
            for i, word in enumerate(words):
                prev_tag = y_pred[i-1] if i > 0 else START_TAG
                for feat, value in featfunc(words, y_true[i], prev_tag, i).iteritems():
                    wsum[feat] += Tinit*value
                    weights[feat] += 1.*value

                for feat, value in featfunc(words, y_pred[i], prev_tag, i).iteritems():
                    wsum[feat] -= Tinit*value
                    weights[feat] -= 1.*value
            tr_err+=sum([y_pred[i] != y_true[i] for i, _ in enumerate(y_pred)])
        Tinit += 1
    # note that i'm computing tr_acc for you, as long as you properly update tr_err
    return weights, wsum, 1.-tr_err / float(sum([len(s) for s,t in inst_generator])), Tinit


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
    T = 1.
    weights = defaultdict(float)
    wsum = defaultdict(float)
    avg_weights = defaultdict(float)

    for i in xrange(N_its):
        # your code here
        weights, wsum, tr_acc_i, T = oneItAvgPerceptron(inst_generator,featfunc,weights,wsum,tagset,T)

        for key, weight in weights.iteritems():
            avg_weights[key] = weight - wsum[key]/T

        confusion = evalTagger(lambda words, alltags: classifierTagger(words,featfunc,avg_weights,tagset),'perc')
        dv_acc[i] = scorer.accuracy(confusion)
        tr_acc[i] = tr_acc_i
        print i,'dev:',dv_acc[i],'train:',tr_acc[i]
    return avg_weights, tr_acc, dv_acc