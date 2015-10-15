import operator
from  constants import *
from collections import defaultdict, Counter
from clf_base import predict, evalClassifier, argmax
import scorer


def trainAvgPerceptron(N_its,inst_generator,labels, outfile, devkey):
    tr_acc = [None]*N_its #holder for training accuracy
    dv_acc = [None]*N_its #holder for dev accuracy
    avg_weights = defaultdict(float)
    weights = defaultdict(float)
    wsum = defaultdict(float)
    tr_tot = 0.
    for i in xrange(N_its):
        weights, wsum, tr_err, tr_tot = oneItAvgPerceptron(inst_generator, weights, wsum, labels, tr_tot) #call your function for a single iteration

        for key, weight in weights.iteritems():
            avg_weights[key] = weights.get(key, 0) - wsum.get(key, 0)/tr_tot

        confusion = evalClassifier(avg_weights, outfile, devkey) #evaluate on dev data
        dv_acc[i] = scorer.accuracy(confusion) #compute accuracy
        tr_acc[i] = 1. - tr_err/float(tr_tot) #compute training accuracy from output
        print i, 'dev: ', dv_acc[i], 'train: ', tr_acc[i]
    return avg_weights, tr_acc, dv_acc

def oneItAvgPerceptron(inst_generator,weights,wsum,labels,Tinit=0):
    tr_err = 0.

    for instance, label in inst_generator:
        Tinit += 1
        label_pred, scores = predict(instance, weights, labels)
        if label_pred != label:
            for word, value in instance.iteritems():
                # Compute running weight (W_T)
                wsum[(label, word)] = wsum.get((label, word), 0) + Tinit * value
                wsum[(label_pred, word)] = wsum.get((label_pred, word), 0) - Tinit * value

                # Update weitgh (w_t)
                weights[(label, word)] = weights.get((label, word), 0) + value
                weights[(label_pred, word)] = weights.get((label_pred, word), 0) - value

            tr_err +=  1

    return weights, wsum, tr_err, Tinit