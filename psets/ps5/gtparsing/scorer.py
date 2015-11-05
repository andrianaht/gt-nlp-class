from collections import defaultdict

def top(confusion, dp, n=10):
    sconfusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:n]

    print dp.reader.pos_dict
    for ((key_tag, key_res), val) in sconfusion[:n]:
        print key_tag, key_res


def getConfusion (keyfile, resfile):
    counts = defaultdict(int)
    with open (keyfile) as kfile, open (resfile) as rfile:
        for keyline in kfile:
            resline = rfile.readline ()
            keyparts = keyline.split ()
            resparts = resline.split ()
            if len (keyparts) > 1 and len (resparts) > 1:
                key_tag = keyparts[6].rstrip()
                res_tag = resparts[6].rstrip()
                counts[tuple((key_tag,res_tag))] += 1
    return (counts)


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