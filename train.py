from experiments.configs.resnet18 import TrainingConfig
from visioner.utils import set_seed, set_cudnn, multi2array
from visioner.models import VisionResNet
from visioner.learners import IMetLearner
from visioner.losses.classification import BinaryFocalLoss
from visioner.metrics.classfication import f2score
from visioner.datasets import IMetDataset
import os
import sys
import random
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import fbeta_score

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

experiment_setting = TrainingConfig.experiment_setting
data_setting = TrainingConfig.data_setting

set_seed(experiment_setting)
set_cudnn(experiment_setting)
while True:  # just for collapsing the code segment...
    # DATA PROCESSING
    train_dir = data_setting['train_dir']
    test_dir = data_setting['test_dir']
    sub_path = data_setting['sub_path']
    train_path = data_setting['train_path']
    label_path = data_setting['label_path']

    sub_df = pd.read_csv(sub_path)
    train_df = pd.read_csv(train_path)
    label_df = pd.read_csv(label_path)

    # protocol
    # return two list of image path (train/test)
    # and a list of image label
    trn_paths = [os.path.join(train_dir, img_name + '.png') for img_name in train_df['id']]
    trn_labels = [[int(idx) for idx in lbl.split(' ')] for lbl in train_df['attribute_ids'].tolist()]
    tst_paths = [os.path.join(test_dir, img_name + '.png') for img_name in sub_df['id']]

    print('[LOG] train num {}, test num {}'.format(len(trn_paths), len(tst_paths)))

    sub_ratio = 0.2

    if not os.path.exists(data_setting['sub_input_path']):
        trn_item_pair = [(tp, tl) for tp, tl in zip(trn_paths, trn_labels)]
        random.shuffle(trn_item_pair)

        sub_trn_item_pair = trn_item_pair[:int(sub_ratio * len(trn_item_pair))]

        sub_trn_paths, sub_trn_labels = [], []
        for stp, stl in sub_trn_item_pair:
            sub_trn_paths.append(stp)
            sub_trn_labels.append(stl)
        pickle.dump([sub_trn_paths, sub_trn_labels], open(data_setting['sub_input_path'], 'wb'))
    else:
        sub_trn_paths, sub_trn_labels = pickle.load(open(data_setting['sub_input_path'], 'rb'))

    x, y = pd.Series(trn_paths), multi2array(trn_labels, class_num=experiment_setting['n_classes'])

    if os.path.exists(data_setting['cv_path']):
        print('[LOG] reading cv-split from {}'.format(data_setting['cv_path']))
        mskf_split = pickle.load(open(data_setting['cv_path'], 'rb'))
    else:
        mskf = MultilabelStratifiedKFold(n_splits=experiment_setting['n_folds'],
                                         random_state=experiment_setting['random_seed'])
        mskf_split = list(mskf.split(x, y))
        pickle.dump(mskf_split, open(data_setting['cv_path'], 'wb'))
        print('[LOG] dump cv-split to {}'.format(data_setting['cv_path']))
    break

fold_idx = 0
print('Something happened..')
for trn_idx, val_idx in mskf_split:
    print('[LOG] val idx top 5 {}'.format(val_idx[:5]))  # fold 0: 2  6  7 12 13
    print('[LOG] fold id {}, train {} val {}'.format(fold_idx, len(trn_idx), len(val_idx)))
    trn_x, val_x = x.iloc[list(trn_idx)], x.iloc[list(val_idx)]
    trn_y, val_y = y[list(trn_idx), :], y[list(val_idx), :]
    train_, val_ = {'trn_x': list(trn_x), 'trn_y': trn_y}, {'val_x': list(val_x), 'val_y': val_y}

    for stage_idx, stage_config in enumerate(TrainingConfig.stages):
        print('[LOG] training stage {}'.format(stage_idx))
        # TODO: use config to select model
        model = VisionResNet(arch=stage_config['model']['arch'],
                             num_classes=experiment_setting['n_classes'],
                             **stage_config['model']['kwargs'])
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=experiment_setting['device_ids'])

        # TODO: use stage_config['optimizer']['type'] to select optimizer
        optimizer = optim.Adam(model.parameters(), **stage_config['optimizer']['kwargs'])
        # TODO: use stage_config['lr_scheduler']['type'] to select lr_scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **stage_config['lr_scheduler']['kwargs'])

        # data loaders
        # TODO: ugly
        train_set = IMetDataset(train_['trn_x'], train_['trn_y'], transform=None, **stage_config['dataset'])
        valid_set = IMetDataset(val_['val_x'], val_['val_y'], transform=None, **stage_config['dataset'])
        train_loader = torch.utils.data.DataLoader(train_set, **stage_config['data_loader']['train_loader'])
        valid_loader = torch.utils.data.DataLoader(valid_set, **stage_config['data_loader']['valid_loader'])

        # TODO: use config
        # TODO: think about it, may need a mapping to map config to real loss functions
        loss_dict = {
            'BCEWithLogitsLoss': {
                'loss_fn': nn.BCEWithLogitsLoss(),
                'weight': 1.
            },
            'FocalLoss': {
                'loss_fn': BinaryFocalLoss(),
                'weight': .1
            }
        }

        # TODO: use config
        metric_dict = {
            'F2': {
                'metric_fn': f2score,
                'args': [],
                'kwargs': {
                    'threshold': 0.1
                }
            }
        }

        learner = IMetLearner(model,
                              optimizer,
                              train_loader,
                              secondary_dataloader=valid_loader,
                              loss_dict=loss_dict,
                              metric_dict=metric_dict,
                              lr_scheduler=scheduler,
                              mode='train')

        learner.fit_one_epoch()

    fold_idx += 1
    break

    # mdl_dir = os.path.join('mdl', configs['prefix'] + '_' + configs['model'], 'fold_' + str(fold_idx))
    # if not os.path.exists(mdl_dir):
    #     os.makedirs(mdl_dir)
    # log_dir = os.path.join('log', configs['prefix'] + '_' + configs['model'], 'fold_' + str(fold_idx))
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    #
    # training(model, optimizer, scheduler, train_, val_, fold_idx)
    # # training_lovasz(model, optimizer, scheduler, train_, val_, fold_idx)
    # fold_idx += 1
