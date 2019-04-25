import torch
from torch import nn

import torch.nn as nn
import torchvision.models as models
from visioner.models.layers import AdaptiveConcatPool2d, Flatten

__all__ = ['VisionResNet']

resnets = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, version, pretrained=True, with_pooling=True, **kwargs):
        super(ResNetFeatureExtractor, self).__init__()
        self.with_pooling = with_pooling
        assert version in resnets
        self.net = resnets[version](pretrained=pretrained, **kwargs)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        if self.with_pooling:
            x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        return x


class ClassificationHead(nn.Module):
    default_pooling = nn.AdaptiveAvgPool2d(output_size=1)

    def __init__(self, num_features=512, num_classes=10, global_pooling_mode='concat', bn_final=False, dropout=0.):
        super(ClassificationHead, self).__init__()

        layers = []
        # global pooling
        assert global_pooling_mode in ['avg', 'max', 'concat'], 'Invalid pooling type for building classification head.'
        global_pooling = ClassificationHead.default_pooling
        if global_pooling_mode == 'avg':
            global_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        elif global_pooling_mode == 'max':
            global_pooling = nn.AdaptiveMaxPool2d(output_size=1)
        elif global_pooling_mode == 'concat':
            global_pooling = AdaptiveConcatPool2d(output_size=1)
        layers.append(global_pooling)
        layers.append(Flatten())

        if dropout > 0.:
            layers.append(torch.nn.Dropout(p=dropout))

        if bn_final:
            layers.append(nn.BatchNorm1d(num_features, momentum=0.01))

        layers.append(nn.Linear(num_features, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class VisionResNet(nn.Module):
    def __init__(self, version, num_classes=10, pretrained=True, with_pooling=True, global_pooling_mode='avg',
                 bn_final=False, dropout=0., mode='classification',
                 **kwargs):
        super(VisionResNet, self).__init__()
        self.version = version
        self.pretrained = pretrained
        self.global_pooling_mode = global_pooling_mode
        self.with_pooling = with_pooling
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.dropout = dropout
        self.bn_final = bn_final
        self.mode = mode
        self.global_pooling = None
        self.body = self._create_body()
        self.head = self._create_head()

    def _create_body(self):
        body = ResNetFeatureExtractor(self.version, pretrained=self.pretrained, with_pooling=self.with_pooling,
                                      **self.kwargs)
        return body

    def _create_head(self):
        # TODO: support other vision tasks
        if self.mode == 'classification':
            head = ClassificationHead(num_features=512, num_classes=self.num_classes,
                                      global_pooling_mode=self.global_pooling_mode, bn_final=self.bn_final,
                                      dropout=self.dropout)

        return head

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
