from constants import OFFSET

def get_token_type_ratio(vocabulary):
    return 1.0 * (len(vocabulary.keys())-1) / sum(vocabulary.values()-vocabulary[OFFSET])

def type_frequency (vocabulary, k):
    return sum([a for a in vocabulary.values() if a == 1])

def unseen_types(first_vocab, second_vocab):
    return set(first_vocab.keys()) - set(second_vocab.keys())
