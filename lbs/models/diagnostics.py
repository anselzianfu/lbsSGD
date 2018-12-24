""" Different diagnostics e.g. accuracy, perplexity """


def accuracy(outputs, targets, size_average=False):
    """ Compute accuracy from softmaxed outputs """
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    if size_average:
        acc = 100. * correct / total
        return acc
    return 100. * correct
