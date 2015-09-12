import operator
from  constants import *
from collections import defaultdict, Counter
from clf_base import predict, evalClassifier
import scorer

argmax = lambda x : max(x.iteritems(), key=operator.itemgetter(1))[0]

def oneItPerceptron(data_generator, weights, labels):
    errors = 0.
    num_insts = 0.

    y_pred = defaultdict(int)
    for instance, label in data_generator:
        num_insts = num_insts + 1
        for all_label in labels:
            for word, value in instance.iteritems():
                y_pred[all_label] += weights.get((all_label, word), 0)*value

        label_pred = argmax(y_pred)

        if label_pred != label:
            for word, value in instance.iteritems():
                weights[(label, word)] = weights.get((label, word), 0) + value
                weights[(label_pred, word)] = weights.get((label_pred, word), 0) - value

            errors = errors + 1

    return weights, errors, num_insts

# this code trains the perceptron for N iterations on the supplied training data
def trainPerceptron(N_its,inst_generator,labels, outfile, devkey):
    tr_acc = [None]*N_its #holder for training accuracy
    dv_acc = [None]*N_its #holder for dev accuracy
    weights = defaultdict(float) 
    for i in xrange(N_its):
        weights,tr_err,tr_tot = oneItPerceptron(inst_generator,weights,labels) #call your function for a single iteration
        confusion = evalClassifier(weights,outfile, devkey) #evaluate on dev data
        dv_acc[i] = scorer.accuracy(confusion) #compute accuracy
        tr_acc[i] = 1. - tr_err/float(tr_tot) #compute training accuracy from output
        print i,'dev: ',dv_acc[i],'train: ',tr_acc[i]
    return weights, tr_acc, dv_acc
