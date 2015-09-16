from scipy.misc import logsumexp #hint
''' your code '''
import operator
from  gtnlplib.constants import ALL_LABELS, OFFSET
from collections import defaultdict, Counter
from gtnlplib.clf_base import evalClassifier, argmax
import numpy as np  
import gtnlplib.scorer as scorer
from gtnlplib.constants import OFFSET, TRAINKEY, DEVKEY

# compute the normalized probability of each label
def computeLabelProbs(instance, weights, labels):
    probs = defaultdict(float)
    for label in labels:
        for word, value in instance.items():
            probs[label] += weights.get((label, word), 0) * value
    normalizer = logsumexp(probs.values())
    return {y: np.exp(probs[y]-normalizer) for y in labels }


def trainLRbySGD(N_its,inst_generator, outfile, devkey, learning_rate=1e-4, regularizer=1e-2):
    weights = defaultdict(float)
    dv_acc = [None]*N_its
    tr_acc = [None]*N_its

    # this block is all to take care of regularization
    ratereg = learning_rate * regularizer
    def regularize(base_feats, t):
        for base_feat in base_feats:
            for label in ALL_LABELS:
                weights[(label, base_feat)] *= (1 - ratereg) ** (t-last_update[base_feat])
            last_update[base_feat] = t

    for it in xrange(N_its):
        tr_err = 0
        last_update = defaultdict(int) # reset, since we regularize at the end of every iteration

        num_inst = len(inst_generator)
        for i, (inst, true_label) in enumerate(inst_generator):
            # apply "just-in-time" regularization to the weights for features in this instance
            regularize(inst, i)
            # compute likelihood gradient from this instance
            probs = computeLabelProbs(inst, weights, ALL_LABELS)

            label_pred = argmax(probs)
            if true_label != label_pred:tr_err += 1

            for word, value in inst.items():
                weights[(true_label, word)] += num_inst * learning_rate * value
                for label in ALL_LABELS:
                    weights[(label, word)] -= num_inst * learning_rate * probs[label] * value

        # regularize all features at the end of each iteration
        regularize([base_feature for label,base_feature in weights.keys()], i)

        dv_acc[it] = scorer.accuracy(evalClassifier(weights, outfile, devkey))
        tr_acc[it] = 1. - tr_err/float(i)
        print it,'dev:',dv_acc[it],'train:',tr_acc[it]
    return weights,tr_acc,dv_acc


def trainLRbyAdaGrad(N_its,inst_generator, outfile, devkey, learning_rate=1e-4, regularizer=1e-2):
    weights = defaultdict(float)
    dv_acc = [None]*N_its
    tr_acc = [None]*N_its

    running_value = defaultdict(float)

    num_inst = len(inst_generator)
    # this block is all to take care of regularization
    ratereg = learning_rate * regularizer
    def regularize(base_feats, t):
        for base_feat in base_feats:
            for label in ALL_LABELS:
                weights[(label, base_feat)] *= (1 - ratereg) ** (t-last_update[base_feat])
            last_update[base_feat] = t

    for it in xrange(N_its):
        tr_err = 0
        last_update = defaultdict(int) # reset, since we regularize at the end of every iteration
        for i, (inst, true_label) in enumerate(inst_generator):
            # apply "just-in-time" regularization to the weights for features in this instance
            regularize(inst, i)
            # compute likelihood gradient from this instance
            probs = computeLabelProbs(inst, weights, ALL_LABELS)

            label_pred = argmax(probs)
            if true_label != label_pred:tr_err += 1

            for word, value in inst.items():
                weights[(true_label, word)] += num_inst * learning_rate * value / running_value.get((true_label, word), 1)
                weights[(label_pred, word)] -= num_inst * probs[label_pred] * learning_rate * value / running_value.get((label_pred, word), 1)
                running_value[(true_label, word)] = value**2

        # regularize all features at the end of each iteration
        regularize([base_feature for label,base_feature in weights.keys()], i)

        dv_acc[it] = scorer.accuracy(evalClassifier(weights, outfile, devkey))
        tr_acc[it] = 1. - tr_err/float(i)
        print it,'dev:',dv_acc[it],'train:',tr_acc[it]
    return weights,tr_acc,dv_acc


def trainLRbyAdaGradMod(N_its,inst_generator, outfile, devkey, learning_rate=1e-4, regularizer=1e-2):
    weights = defaultdict(float)
    dv_acc = [None]*N_its
    tr_acc = [None]*N_its

    running_value = defaultdict(float)

    num_inst = len(inst_generator)
    # this block is all to take care of regularization
    ratereg = learning_rate * regularizer
    def regularize(base_feats, t):
        for base_feat in base_feats:
            for label in ALL_LABELS:
                weights[(label, base_feat)] *= (1 - ratereg) ** (t-last_update[base_feat])
            last_update[base_feat] = t

    for it in xrange(N_its):
        tr_err = 0
        last_update = defaultdict(int) # reset, since we regularize at the end of every iteration
        for i, (inst, true_label) in enumerate(inst_generator):
            # apply "just-in-time" regularization to the weights for features in this instance
            regularize(inst, i)
            # compute likelihood gradient from this instance
            probs = computeLabelProbs(inst, weights, ALL_LABELS)

            label_pred = argmax(probs)
            if true_label != label_pred:tr_err += 1

            for word, value in inst.items():
                weights[(true_label, word)] += num_inst * learning_rate * value / running_value.get((true_label, word), 1)
                weights[(label_pred, word)] -= num_inst * learning_rate * value / running_value.get((label_pred, word), 1)
                running_value[(true_label, word)] = value**2

        # regularize all features at the end of each iteration
        regularize([base_feature for label,base_feature in weights.keys()], i)

        dv_acc[it] = scorer.accuracy(evalClassifier(weights, outfile, devkey))
        tr_acc[it] = 1. - tr_err/float(i)
        print it,'dev:',dv_acc[it],'train:',tr_acc[it]
    return weights,tr_acc,dv_acc


def regularization_using_grid_search(alphas, netas, N_its,inst_generator, outfile, devkey, learning_rate=1e-4, regularizer=1e-2, tr_outfile='logreg.alpha.tr.txt', dv_outfile='logreg.alpha.dv.txt'):
    tr_accs = []
    dv_accs = []
    # Choose your alphas here
    weights_log_reg_alphas = dict()
    for alpha in alphas:
        for neta in netas:
            weights_log_reg_alphas[(alpha, neta)] = trainLRbySGD(N_its,inst_generator, outfile, devkey, learning_rate=neta, regularizer=alpha)
            confusion = evalClassifier(weights_log_reg_alphas[(alpha, neta)],tr_outfile,TRAINKEY)
            tr_accs.append(scorer.accuracy(confusion))
            confusion = evalClassifier(weights_log_reg_alphas[(alpha, neta)],dv_outfile,DEVKEY)
            dv_accs.append(scorer.accuracy(confusion))
    return weights_log_reg_alphas, tr_accs, dv_accs