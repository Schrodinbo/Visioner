NUM_CLASS = 1103


class TrainingConfig:
    experiment_name = 'Sample'
    experiment_setting = {
        'type': 'classification',
        'target_type': 'multiclass+multilabel',
        'n_classes': NUM_CLASS,
        'deterministic': False,
        'benchmark': False,
        'use_cuda': True,
        'device_ids': [0, 1],
        'random_seed': 2099,
        'cv': True,
        'n_folds': 5,
    }
    data_setting = {
        'train_dir': './input/train',
        'test_dir': './input/test',
        'sub_path': './input/sample_submission.csv',
        'train_path': './input/train.csv',
        'label_path': './input/labels.csv',
        'sub_input_path': './sub_input.pkl',
        'cv_path': './cv_splits.pickle'
    }
    stages = [{

        'model': {
            'arch': 'resnet18',
            'kwargs': {
                'pretrained': True,
                'with_pooling': True,
                'global_pooling_mode': 'avg',
                'dropout': 0.,
                'mode': 'logits'
            }
        },
        'dataset': {
            'img_size': (128, 128)
        },
        'data_loader': {
            'train_loader': {
                'num_workers': 0,
                'batch_size': 24,
                'shuffle': True,
                'pin_memory': True
            },
            'valid_loader': {
                'num_workers': 8,
                'batch_size': 24,
                'shuffle': False,
                'pin_memory': True
            },
            'test_loader': {
                'num_workers': 8,
                'batch_size': 24,
                'shuffle': False,
                'pin_memory': True
            }
        },
        'loss': [
            {
                'type': 'BCEWithLogits',
                'kwargs': {}
            }
        ],
        'metrics': [
            {
                'type': 'F2',
                'kwargs': {}
            }
        ],
        'learner': {
            'batch_size': 64,
            'threshold': 0.1,
            'start_epoch': 0,
            'train_epoch': 30
        },
        'optimizer': {
            'type': 'Adam',
            'kwargs': {
                'lr': 3e-4
            }
        },
        'lr_scheduler': {
            'type': 'ReduceLROnPlateau',
            'kwargs': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 2,
                'min_lr': 1e-6,
                'verbose': True
            }
        },
        'logger': {

        }
    }]
