from visioner.base import BaseLearner
from visioner.utils import AverageMeter
from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid

import numpy as np


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

    def get_meters(self):
        self.train_loss_meters = {name: AverageMeter() for name in self.loss_dict}
        self.train_metric_meters = {name: AverageMeter() for name in self.metric_dict}
        if self.secondary_dataloader:
            self.valid_loss_meters = {name: AverageMeter() for name in self.loss_dict}
            self.valid_metric_meters = {name: AverageMeter() for name in self.metric_dict}

    def log_meters(self):
        train_meters = [self.train_loss_meters, self.train_metric_meters]
        for meters in train_meters:
            for meter_name in meters:
                self.writer.add_scalar(f'train/{meter_name}', meters[meter_name].val, self.iter)

        if self.secondary_dataloader:
            valid_meters = [self.valid_loss_meters, self.valid_metric_meters]
            for meters in valid_meters:
                for meter_name in meters:
                    self.writer.add_scalar(f'valid/{meter_name}', meters[meter_name].val, self.iter)
        # writer.add_scalar('val/loss', val_loss, step)


    def train(self):
        self.model.train(True)
        for epoch in range(self.start_epoch, self.train_epoch):
            self.fit_one_epoch()
            self.epoch += 1

    def validate(self):
        preds, labels = [], []
        print('=====================Valid=====================')
        for batch_idx, data in enumerate(self.secondary_dataloader):
            losses_report, valid_preds, valid_labels = self.forward_one_batch(data)
            for loss in losses_report:
                self.valid_loss_meters[loss].update(losses_report[loss].detach().cpu().item())
            preds.append(valid_preds)
            labels.append(valid_labels)
            # TODO: testing only
            if batch_idx > 10:
                break
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        metrics_report = self.evaluate_metrics(preds, labels)
        for metric in metrics_report:
            self.valid_metric_meters[metric].update(metrics_report[metric])
        #print(losses_report)
        #print(metrics_report)
        print(self.valid_loss_meters)
        print(self.valid_metric_meters)

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

    def _create_meters(self):
        pass

    def fit_one_epoch(self):
        self.model.train(True)
        self.get_meters()
        preds, labels = [], []
        for batch_idx, data in enumerate(self.primary_dataloader):
            losses_report, train_preds, train_labels = self.forward_one_batch(data)
            for loss in losses_report:
                self.train_loss_meters[loss].update(losses_report[loss].detach().cpu().item())

            preds.append(train_preds)
            labels.append(train_labels)
            self.iter += 1
            if self.iter % 25 == 0:
                print('batch done', self.iter)
                print('=====================Train=====================')
                # TODO: track train
                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                metrics_report = self.evaluate_metrics(preds, labels)
                for metric in metrics_report:
                    self.train_metric_meters[metric].update(metrics_report[metric])
                #print(losses_report)
                #print(metrics_report)
                print(self.train_loss_meters)
                print(self.train_metric_meters)
                preds, labels = [], []

                if self.secondary_dataloader:
                    # TODO: track valid
                    self.model.train(False)
                    self.validate()
                    self.model.train(True)
                self.log_meters()
                self.get_meters()

    def forward_one_batch(self, data):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        outputs = self.model(inputs)
        logits = outputs.sigmoid()
        losses_report = self.compute_losses(outputs, labels)

        # TODO: so ugly
        if self.model.training:
            losses = [losses_report[loss_name] * self.loss_dict[loss_name]['weight'] for loss_name in self.loss_dict]
            backward_losses = losses[0]
            for i in range(1, len(losses)):
                backward_losses += losses[i]

            backward_losses.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # print(backward_losses.detach().cpu().item())

        return losses_report, logits.detach().cpu().numpy(), labels.detach().cpu().numpy()

    def evaluate_metrics(self, predicted, target):
        metrics_report = {}
        for metric_name in self.metric_dict:
            metric_fn = self.metric_dict[metric_name]['metric_fn']
            args = self.metric_dict[metric_name]['args']
            metric_kwargs = self.metric_dict[metric_name]['kwargs']
            metrics_report[metric_name] = metric_fn(predicted, target, *args, **metric_kwargs)
        return metrics_report

    def compute_losses(self, predicted, target):
        losses_report = {}
        for loss_name in self.loss_dict:
            loss_fn = self.loss_dict[loss_name]['loss_fn']
            # kwargs = self.loss_dict[loss_name]['kwargs']
            losses_report[loss_name] = loss_fn(predicted, target)
        return losses_report

    def save_checkpoint(self):
        checkpoint = {
            'model_weights': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': None
        }
        # save_path = os.path.join('mdl', configs['prefix'] + '_' + configs['model'], 'fold_' + str(fold_idx),
        #                          'weights-%d-%d-loss-[%.4f]-f2-[%.4f].pth' % (epoch, batch_cnt, val_loss, val_f2))
        # torch.save(model.state_dict(), save_path)
