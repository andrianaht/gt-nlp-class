import numpy as np
from collections import defaultdict

# Implement for deliverable 2a
def CPT (instances, htag):
    """ Accepts instances which is a list and a tag index.
        Computes the conditional probability of modifier given the head tag.

        params:
        instances: list
        htag: integer

        returns:
        output: Dict - where key is a tag and the value is probability.
    """
    output = defaultdict(float)
    for instance in instances:
        for i, head in enumerate(instance.heads):
            if instance.pos[head] == htag and i != 0:
                output[instance.pos[i]] += 1
    return {key: value/sum(output.values()) for key, value in output.iteritems()}


def entropy (distr):
    """ Calculates the entropy of a given distribution """
    return -sum([p * np.log(p) for p in distr.values()])
