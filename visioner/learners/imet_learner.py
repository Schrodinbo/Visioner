from visioner.base import BaseLearner
from visioner.utils import AverageMeter
from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid

import numpy as np



"""
        kwargs = {
            'secondary_dataloader': secondary_dataloader,
            'loss_fn': loss_fn,
            'loss_fn_weights': loss_fn_weights,
            'metrics': metrics,
            'lr_scheduler': lr_scheduler,
            'mode': mode
        }
"""


class IMetLearner(BaseLearner):
    def __init__(self,
                 model,
                 optimizer,
                 primary_dataloader,
                 **kwargs
                 ):
        super().__init__(model, optimizer, primary_dataloader, **kwargs)
        self.epoch = 0
        self.iter = 0
        self.writer = SummaryWriter('./experiments/logs')

    def train(self):
        self.model.train(True)
        for epoch in range(self.start_epoch, self.train_epoch):
            self.fit_one_epoch()
            self.epoch += 1

    def validate(self):
        self.model.train(False)
        val_loss = AverageMeter()
        val_focal = AverageMeter()
        val_label, val_pred = [], []
        for batch_cnt_val, data_val in enumerate(self.secondary_dataloader):
            inputs, labels = data_val
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = self.model(inputs)
            logits = sigmoid(outputs)
            # loss = criterion(outputs, labels)
            # focal_metric = focal_loss(outputs, labels)
            val_loss.update(loss.data.item())
            val_focal.update(focal_metric.data.item())

            val_label.append(labels.cpu().numpy())
            val_pred.append(logits.cpu().numpy())
        val_label = np.concatenate(val_label, axis=0)
        val_pred = np.concatenate(val_pred, axis=0)

        self.evaluate_metrics(val_pred, val_label)
        val_f2 = f2score(val_label, val_pred)

        self.writer.add_scalar('val/loss', val_loss, self.iter)
        self.writer.add_scalar('val/f2', val_f2, self.iter)
        self.writer.add_scalar('val/focal', val_focal, self.iter)

        # print('--' * 30)
        # print('{}-epoch-{}-step-{}-val_loss-{:.4f}-val_f2@{}-{:.4f}-val_focal-{:.4f}'.format(dt(), epoch, step,
        #                                                                                      val_loss,
        #                                                                                      configs['thr'],
        #                                                                                      val_f2, val_focal))
        # writer.add_scalar('val/loss', val_loss, step)
        # writer.add_scalar('val/f2', val_f2, step)
        # writer.add_scalar('val/focal', val_focal, step)
        #
        # save_path = os.path.join('mdl', configs['prefix'] + '_' + configs['model'], 'fold_' + str(fold_idx),
        #                          'weights-%d-%d-loss-[%.4f]-f2-[%.4f].pth' % (
        #                              epoch, batch_cnt, val_loss, val_f2))
        # torch.save(model.state_dict(), save_path)
        # print('saved model to %s' % (save_path))
        # print('--' * 30)

    def fit_one_epoch(self):
        for batch_idx, data in enumerate(self.primary_dataloader):
            self.fit_one_batch(data)
            self.iter += 1
            print('batch done', self.iter)
            if self.iter % 500 == 0:
                self.validate()

    def fit_one_batch(self, data):
        self.model.train(True)
        inputs, labels = data

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        logits = sigmoid(outputs)

        losses = self.compute_losses(outputs, labels)
        # TODO: ugly
        losses = [losses[i] * self.loss_fn_weights[i] for i in range(len(losses))]
        backward_losses = losses[0]
        for i in range(1, len(losses)):
            backward_losses += losses[i]

        backward_losses.backward()
        self.optimizer.step()
        print(backward_losses.detach().cpu().item())
        if self.iter % 500 == 0 and self.writer:
            self.writer.add_scalar('train/loss', backward_losses.detach().cpu().item(), self.iter)
            # self.writer.add_scalar('train/f2', f2_value,  self.iter)
            # self.writer.add_scalar('train/focal', focal_value,  self.iter)

        return logits

    def evaluate_metrics(self, predicted, target):
        metrics = {}
        for metric in self.metrics:
            val = metric(predicted, target)
            metrics[metric.__classname__] = val
        return metrics

    def compute_losses(self, predicted, target):
        losses = []
        for loss_fn in self.loss_fn:
            loss = loss_fn(predicted, target)
            losses.append(loss)
        return losses

    def save_model(self):
        save_path = os.path.join('mdl', configs['prefix'] + '_' + configs['model'], 'fold_' + str(fold_idx),
                                 'weights-%d-%d-loss-[%.4f]-f2-[%.4f].pth' % (epoch, batch_cnt, val_loss, val_f2))
        torch.save(model.state_dict(), save_path)
