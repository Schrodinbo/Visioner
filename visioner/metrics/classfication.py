# accuracy
# recall
# precision
# fbeta (f1, f2..., micro and macro..)
# auc


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
