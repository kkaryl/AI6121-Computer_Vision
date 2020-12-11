

__all__ = ['AverageMeter', 'get_error', 'print_numbers_acc']
        
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.summ = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.summ += val * n
        self.count += n
        self.avg = self.summ / self.count

def get_error(scores, labels):
    """
    https://github.com/xbresson/AI6103_2020
    """
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return num_matches.float()/bs 

def print_numbers_acc(accs, numbers=range(10)):
    assert len(accs) == len(numbers)
    assert type(accs[0]) == AverageMeter
    for t, a in zip(accs, numbers):
        print(f"{a}: {t.avg}")
    return {a:t.avg.item() for t, a in zip(accs, numbers)}