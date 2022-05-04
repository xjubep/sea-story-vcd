class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def safe_ratio(a, b, eps=1e-12):
    return a / (b + eps)


def fscore(prec, rec, eps=1e-12):
    return 2 * prec * rec / (prec + rec + eps)
