from collections import defaultdict, OrderedDict


# ( True, Pred )
def getConfusion (keyfile, resfile):
    with open (keyfile) as kfile, open (resfile) as rfile:
        kfiles = kfile.read().split('\n''\n')
        rfiles = rfile.read().split('\n''\n')
        confusion = defaultdict(int)
        for i, ksent1 in enumerate(kfiles):
            ksent = ksent1.split('\n')
            rsent = rfiles[i].split('\n')
            for j, keyline in enumerate(ksent):
                resline = rsent[j]
                keyparts = keyline.split ()
                resparts = resline.split ()
                if len (keyparts) > 1 and len (resparts) > 1:
                    key_tag = int(keyparts[6].rstrip())
                    res_tag = int(resparts[6].rstrip())
                    key_tag = ksent[key_tag-1].split()[3].rstrip() if key_tag > 0 else keyparts[3].rstrip()
                    res_tag = rsent[res_tag-1].split()[3].rstrip() if res_tag > 0 else resparts[3].rstrip()
                    confusion[tuple((key_tag,res_tag))] += 1

    return confusion


def getStatistics(keyfile, resfile, n=10):
    confusion = getConfusion(keyfile, resfile)
    correct = defaultdict(float)
    error  = defaultdict(float)
    correctDist = defaultdict(float)
    errorDist = defaultdict(float)
    # for tag in alltags:
    for (true, pred), value in confusion.iteritems():
        if pred == true:
            correct[true] += value
        else:
            error[true] += value

    for tag, value in correct.iteritems():
        correctDist[tag] = correct[tag]/(correct[tag]+error[tag])

    for tag, value in error.iteritems():
        errorDist[tag] = error[tag]/(correct[tag]+error[tag])

    best = OrderedDict(sorted(correctDist.items(), key=lambda t: t[1], reverse=True)[:n])
    worst = OrderedDict(sorted(errorDist.items(), key=lambda t: t[1], reverse=True)[:n])

    print '----- Best prediction -----'
    for key, value in best.iteritems():
        print key, value, correct[key]

    print
    print '---- Worst Prediction ----'
    for key, value in worst.iteritems():
        print key, value, error[key]
    return best, worst


def printScoreMessage(counts):
    true_pos = 0
    total = 0

    keyclasses = set([x[0] for x in counts.keys()])
    resclasses = set([x[1] for x in counts.keys()])
    print "%d classes in key: %s" % (len(keyclasses),keyclasses)
    print "%d classes in response: %s" % (len(resclasses),resclasses)
    print "confusion matrix"
    print "key\\response:\t"+"\t".join(resclasses)
    for i,keyclass in enumerate(keyclasses):
        print keyclass+"\t\t",
        for j,resclass in enumerate(resclasses):
            c = counts[tuple((keyclass,resclass))]
            #countarr[i,j] = c
            print "{}\t".format(c),
            total += float(c)
            if resclass==keyclass:
                true_pos += float(c)
        print ""
    print "----------------"
    print "accuracy: %.4f = %d/%d\n" % (true_pos / total, true_pos,total)