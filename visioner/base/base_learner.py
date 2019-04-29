from abc import ABC, abstractmethod


class BaseLearner(ABC):
    def __init__(self,
                 model,
                 optimizer,
                 primary_dataloader,
                 secondary_dataloader=None,
                 loss_dict=None,
                 metric_dict=None,
                 lr_scheduler=None,
                 start_epoch=0,
                 train_epoch=2,
                 mode='train'):
        self.model = model
        self.optimizer = optimizer
        self.primary_dataloader = primary_dataloader
        self.secondary_dataloader = secondary_dataloader
        self.loss_dict = loss_dict
        self.metric_dict = metric_dict
        self.start_epoch = start_epoch
        self.train_epoch = train_epoch
        self.mode = mode
        self.lr_scheduler = lr_scheduler

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def fit_one_epoch(self):
        pass

    @abstractmethod
    def forward_one_batch(self, data):
        pass

    @abstractmethod
    def evaluate_metrics(self, predicted, target):
        pass

    @abstractmethod
    def compute_losses(self, predicted, target):
        pass
