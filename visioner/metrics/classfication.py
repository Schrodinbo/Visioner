# accuracy
# recall
# precision
# fbeta (f1, f2..., micro and macro..)
# auc
import numpy as np


def f2score(y_pred, y_true, beta_f2=2, threshold=0.1):
    y_pred = np.array((y_pred > threshold), dtype=np.int8)

    assert y_true.shape[0] == y_pred.shape[0]
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    p = tp / (tp + fp + 1e-15)
    r = tp / (tp + fn + 1e-15)
    f2 = (1 + beta_f2 ** 2) * p * r / (p * beta_f2 ** 2 + r + 1e-15)

    return f2


def fbeta(y_pred, y_true, threshold=0.1, beta=2, eps=1e-9, sigmoid=True):
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean()


def accuracy(input, targs):
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    return (input == targs).float().mean()


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.byte()).float().mean()


def top_k_accuracy(input, targs, k=5):
    input = input.topk(k=k, dim=-1)[1]
    targs = targs.unsqueeze(dim=-1).expand_as(input)
    return (input == targs).max(dim=-1)[0].float().mean()


def error_rate(input, targs):
    return 1 - accuracy(input, targs)
