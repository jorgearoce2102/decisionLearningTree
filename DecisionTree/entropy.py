import math

log = math.log

def get_boolean_entropy(q):
    """Function that returns the boolean entropy
    args
    q: probability of a  Boolan random variable that is true
    """

    return -(q * log(q,2) + (1 - q) * log((1 - q),2))
