# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision import models, transforms


class _SeDenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_SeDenseLayer, self).__init__()

        # original dense modules
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

        # SE modules
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(num_input_features, num_input_features // 16, kernel_size=1, bias=False)
        self.relu_SE = nn.ReLU(inplace=True)
        self.conv_up = nn.Conv2d(num_input_features // 16, num_input_features, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x0 = self.norm1(x)

        x1 = self.global_pooling(x0)
        x1 = self.conv_down(x1)
        x1 = self.relu_SE(x1)
        x1 = self.conv_up(x1)
        x1 = self.sig(x1)

        x0 = x1 * x0
        x0 = self.relu1(x0)
        x0 = self.conv1(x0)
        x0 = self.norm2(x0)
        x0 = self.relu2(x0)
        new_features = self.conv2(x0)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _SeDenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_SeDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _SeDenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('se_denselayer%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # original transition modules
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # SE modules
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(num_input_features, num_input_features // 16, kernel_size=1, bias=False)
        self.relu_SE = nn.ReLU(inplace=True)
        self.conv_up = nn.Conv2d(num_input_features // 16, num_input_features, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x0 = self.norm(x)

        x1 = self.global_pooling(x0)
        x1 = self.conv_down(x1)
        x1 = self.relu_SE(x1)
        x1 = self.conv_up(x1)
        x1 = self.sig(x1)

        x0 = x1 * x0
        x0 = self.relu(x0)
        x0 = self.conv(x0)
        x0 = self.pool(x0)

        return x0


class _Transition_nopooling(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition_nopooling, self).__init__()
        # original transition modules
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # SE modules
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(num_input_features, num_input_features // 16, kernel_size=1, bias=False)
        self.relu_SE = nn.ReLU(inplace=True)
        self.conv_up = nn.Conv2d(num_input_features // 16, num_input_features, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x0 = self.norm(x)

        x1 = self.global_pooling(x0)
        x1 = self.conv_down(x1)
        x1 = self.relu_SE(x1)
        x1 = self.conv_up(x1)
        x1 = self.sig(x1)

        x0 = x1 * x0
        x0 = self.relu(x0)
        x0 = self.conv(x0)
        # x0 = self.pool(x0)

        return x0


class SeDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=6):

        super(SeDenseNet, self).__init__()

        self.model_p = models.densenet121(pretrained=True)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _SeDenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('se_denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 2:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            elif i == len(block_config) - 2:
                trans = _Transition_nopooling(num_input_features=num_features, num_output_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.final_SE = nn.Sequential(OrderedDict([
            ('global_pooling', nn.AdaptiveAvgPool2d(1)),
            ('conv_down', nn.Conv2d(num_features, num_features // 16, kernel_size=1, bias=False)),
            ('relu_SE', nn.ReLU(inplace=True)),
            ('conv_up', nn.Conv2d(num_features // 16, num_features, kernel_size=1, bias=False)),
            ('sig', nn.Sigmoid())
        ]))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal(m.weight.data)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

        # init with densenet121 model parameters
        se_dict = self.state_dict()
        param_dict = self.model_p.state_dict()
        se_dict['features.conv0.weight'] = param_dict['features.conv0.weight']
        se_dict['features.norm0.weight'] = param_dict['features.norm0.weight']
        se_dict['features.norm0.bias'] = param_dict['features.norm0.bias']
        for i, num_layer in enumerate(block_config):
            if i != len(block_config) - 1:
                for k in range(num_layer):
                    se_dict['features.se_denseblock{}.se_denselayer{}.norm1.weight'.format(i + 1, k + 1)] = \
                        param_dict['features.denseblock{}.denselayer{}.norm1.weight'.format(i + 1, k + 1)]

                    se_dict['features.se_denseblock{}.se_denselayer{}.norm1.bias'.format(i + 1, k + 1)] = \
                        param_dict['features.denseblock{}.denselayer{}.norm1.bias'.format(i + 1, k + 1)]

                    se_dict['features.se_denseblock{}.se_denselayer{}.conv1.weight'.format(i + 1, k + 1)] = \
                        param_dict['features.denseblock{}.denselayer{}.conv1.weight'.format(i + 1, k + 1)]

                    se_dict['features.se_denseblock{}.se_denselayer{}.norm2.weight'.format(i + 1, k + 1)] = \
                        param_dict['features.denseblock{}.denselayer{}.norm2.weight'.format(i + 1, k + 1)]

                    se_dict['features.se_denseblock{}.se_denselayer{}.norm2.bias'.format(i + 1, k + 1)] = \
                        param_dict['features.denseblock{}.denselayer{}.norm2.bias'.format(i + 1, k + 1)]

                    se_dict['features.se_denseblock{}.se_denselayer{}.conv2.weight'.format(i + 1, k + 1)] = \
                        param_dict['features.denseblock{}.denselayer{}.conv2.weight'.format(i + 1, k + 1)]

                    nn.init.kaiming_normal_(
                        se_dict['features.se_denseblock{}.se_denselayer{}.conv_down.weight'.format(i + 1, k + 1)])

                    nn.init.kaiming_normal_(
                        se_dict['features.se_denseblock{}.se_denselayer{}.conv_up.weight'.format(i + 1, k + 1)])

                    # ---------modified by chun-----------------#
                if i != len(block_config) - 2:
                    se_dict['features.transition{}.norm.weight'.format(i + 1)] = param_dict[
                        'features.transition{}.norm.weight'.format(i + 1)]

                    se_dict['features.transition{}.norm.bias'.format(i + 1)] = param_dict[
                        'features.transition{}.norm.bias'.format(i + 1)]

                    se_dict['features.transition{}.conv.weight'.format(i + 1)] = param_dict[
                        'features.transition{}.conv.weight'.format(i + 1)]

                    nn.init.kaiming_normal_(se_dict['features.transition{}.conv_down.weight'.format(i + 1)])

                    nn.init.kaiming_normal_(se_dict['features.transition{}.conv_up.weight'.format(i + 1)])
                elif i == len(block_config) - 2:
                    se_dict['features.transition{}.norm.weight'.format(i + 1)] = param_dict[
                        'features.transition{}.norm.weight'.format(i + 1)]

                    se_dict['features.transition{}.norm.bias'.format(i + 1)] = param_dict[
                        'features.transition{}.norm.bias'.format(i + 1)]
                    nn.init.kaiming_normal_(se_dict['features.transition{}.conv.weight'.format(i + 1)])
                    nn.init.kaiming_normal_(se_dict['features.transition{}.conv_down.weight'.format(i + 1)])

                    nn.init.kaiming_normal_(se_dict['features.transition{}.conv_up.weight'.format(i + 1)])


            # if i != len(block_config) - 1:
            # se_dict['features.transition{}.norm.weight'.format(i + 1)] = param_dict[
            #     'features.transition{}.norm.weight'.format(i + 1)]
            #
            # se_dict['features.transition{}.norm.bias'.format(i + 1)] = param_dict[
            #     'features.transition{}.norm.bias'.format(i + 1)]
            #
            # se_dict['features.transition{}.conv.weight'.format(i + 1)] = param_dict[
            #     'features.transition{}.conv.weight'.format(i + 1)]
            #
            # nn.init.kaimi   ng_normal_(se_dict['features.transition{}.conv_down.weight'.format(i + 1)])
            #
            # nn.init.kaiming_normal_(se_dict['features.transition{}.conv_up.weight'.format(i + 1)])

            elif i == len(block_config) - 1:
                for k in range(num_layer):
                    # nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.norm1.weight'.format(i + 1, k + 1)])
                    # nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.norm1.bias'.format(i + 1, k + 1)])
                    nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.conv1.weight'.format(i + 1, k + 1)])
                    # nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.norm2.weight'.format(i + 1, k + 1)])
                    # nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.norm2.bias'.format(i + 1, k + 1)])
                    nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.conv2.weight'.format(i + 1, k + 1)])
                    nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.conv_down.weight'.format(i + 1, k + 1)])
                    nn.init.kaiming_normal_(se_dict['features.se_denseblock{}.se_denselayer{}.conv_up.weight'.format(i + 1, k + 1)])


        se_dict['features.norm5.weight'].fill_(1)
        se_dict['features.norm5.bias'].zero_()
        nn.init.kaiming_normal_(se_dict['final_SE.conv_down.weight'])
        nn.init.kaiming_normal_(se_dict['final_SE.conv_up.weight'])
        se_dict['classifier.bias'].zero_()

        self.load_state_dict(se_dict)

    def forward(self, x):
        x0 = self.features(x)
        x1 = self.final_SE(x0)

        features = x1 * x0
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
