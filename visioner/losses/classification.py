import torch
import torch.nn.functional as F


def BinaryFocalLoss(gamma=2):
    def func(input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss
        return loss.mean()

    return func


def f2_loss(y_true, y_pred, epsilon=1e-12, beta_f2=2):
    tp = torch.sum(y_true * y_pred)
    tn = torch.sum((1 - y_true) * (1 - y_pred))
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    # f = 2*p*r / (p+r+epsilon)
    f = (1 + beta_f2 ** 2) * p * r / (p * beta_f2 ** 2 + r + epsilon)
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return torch.mean(f)
