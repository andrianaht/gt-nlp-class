from constants import OFFSET

def get_token_type_ratio(vocabulary):
    return 1.0 * (sum(vocabulary.values())) / (len(vocabulary.keys()))

def type_frequency (vocabulary, k):
    return sum([a for a in vocabulary.values() if a == 1])

def unseen_types(first_vocab, second_vocab):
    return len(set(second_vocab.keys()) - set(first_vocab.keys()))