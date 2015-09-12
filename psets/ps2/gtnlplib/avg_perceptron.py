import operator
from  constants import *
from collections import defaultdict, Counter
from clf_base import predict, evalClassifier
import scorer


argmax = lambda x : max(x.iteritems(), key=operator.itemgetter(1))[0]

def trainAvgPerceptron(N_its,inst_generator,labels, outfile, devkey):
    tr_acc = [None]*N_its #holder for training accuracy
    dv_acc = [None]*N_its #holder for dev accuracy
    avg_weights = defaultdict(float)
    wsum = defaultdict(float)
    tr_tot = 0.
    for i in xrange(N_its):
        avg_weights, wsum, tr_err, tr_tot = oneItAvgPerceptron(inst_generator, avg_weights, wsum, labels, tr_tot) #call your function for a single iteration
        confusion = evalClassifier(avg_weights, outfile, devkey) #evaluate on dev data
        dv_acc[i] = scorer.accuracy(confusion) #compute accuracy
        tr_acc[i] = 1. - (i+1)*tr_err/float(tr_tot) #compute training accuracy from output
        print i, 'dev: ', dv_acc[i], 'train: ', tr_acc[i]
    return avg_weights, tr_acc, dv_acc

def oneItAvgPerceptron(inst_generator,weights,wsum,labels,Tinit=0):
    tr_err = 0.
    num_insts = 0.

    y_pred = defaultdict(int)
    for instance, label in inst_generator:
        num_insts = num_insts + 1
        for all_label in labels:
            for word, value in instance.iteritems():
                y_pred[all_label] += weights.get((all_label, word), 0)*value

        label_pred = argmax(y_pred)

        if label_pred != label:
            for word, value in instance.iteritems():
                # Compute wsum
                wsum[(label, word)] = wsum.get((label, word), 0) + Tinit * value

                # Update weitgh_t
                weights[(label, word)] = weights.get((label, word), 0) + value
                weights[(label_pred, word)] = weights.get((label_pred, word), 0) - value

                # Update
                weights[(label, word)] = weights.get((label, word), 0) - wsum.get((label, word), 0)/Tinit
                weights[(label_pred, word)] = weights.get((label_pred, word), 0) - wsum.get((label_pred, word), 0)/Tinit


            tr_err = tr_err + 1

        Tinit = Tinit + 1


    return weights, wsum, tr_err, Tinit
