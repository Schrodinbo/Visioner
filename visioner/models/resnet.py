import torch.nn as nn
import torchvision.models as models

from torchvision.models.resnet import BasicBlock, Bottleneck

from visioner.models.modules import LogitsHead

__all__ = ['VisionResNet']

resnets = {
    'resnet18': {
        'model': models.resnet18,
        'num_features': 512 * BasicBlock.expansion
    },
    'resnet34': {
        'model': models.resnet34,
        'num_features': 512 * BasicBlock.expansion
    },
    'resnet50': {
        'model': models.resnet50,
        'num_features': 512 * Bottleneck.expansion
    },
    'resnet101': {
        'model': models.resnet101,
        'num_features': 512 * Bottleneck.expansion
    },
    'resnet152': {
        'model': models.resnet152,
        'num_features': 512 * Bottleneck.expansion
    },
}


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True, with_pooling=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.with_pooling = with_pooling
        assert arch in resnets, 'Only ResNet18/34/50/101/152 are supported at this moment.'
        self.resnet = resnets[arch]['model'](pretrained=pretrained)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        if self.with_pooling:
            x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x


class VisionResNet(nn.Module):
    def __init__(self,
                 arch,
                 num_classes=10,
                 pretrained=True,
                 with_pooling=True,
                 global_pooling_mode='avg',
                 dropout=0.,
                 mode='logits',
                 predefined_out_layer=None,
                 predefined_head=None
                 ):
        super(VisionResNet, self).__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.global_pooling_mode = global_pooling_mode
        self.with_pooling = with_pooling
        self.num_classes = num_classes
        self.dropout = dropout
        self.mode = mode
        self.global_pooling = None
        self.predefined_out_layer = predefined_out_layer

        self.body = self._create_body()
        self.head = self._create_head() if not predefined_head else predefined_head

    def _create_body(self):
        body = ResNetFeatureExtractor(
            self.arch,
            pretrained=self.pretrained,
            with_pooling=self.with_pooling
        )
        return body

    def _create_head(self):
        # TODO: support other vision tasks
        if self.mode == 'logits':
            head = LogitsHead(
                num_features=resnets[self.arch]['num_features'],
                num_classes=self.num_classes,
                global_pooling_mode=self.global_pooling_mode,
                dropout=self.dropout,
                predefined_out_layer=self.predefined_out_layer
            )
        else:
            raise NotImplementedError('Currently only the MLP head is supported.')

        return head

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from visioner.models.net_tests import check_resnet

    model = VisionResNet('resnet18')
    check_resnet(model)

    model = VisionResNet('resnet34')
    check_resnet(model)

    model = VisionResNet('resnet50')
    check_resnet(model)

    model = VisionResNet('resnet101')
    check_resnet(model)

    model = VisionResNet('resnet152')
    check_resnet(model)
