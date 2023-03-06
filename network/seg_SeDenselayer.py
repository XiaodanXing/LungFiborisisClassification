# encoding:utf-8
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision import models, transforms
import torch.optim as optim
import copy


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


class FirstLayer(nn.Module):
    def __init__(self, num_init_features=64, in_channel=1):
        super(FirstLayer, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

    def forward(self, input):
        out = self.features(input)
        return out


class Common_SeDensenet_parts(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=6):
        super(Common_SeDensenet_parts, self).__init__()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _SeDenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('se_denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

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

    def forward(self, x):
        x0 = self.features(x)
        x1 = self.final_SE(x0)

        features = x1 * x0
        out1 = features
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)
        out = self.classifier(out)
        weight = self.classifier.weight
        return out, out1, weight


class Resemble_Network(nn.Module):
    def __init__(self, in_channel):
        super(Resemble_Network, self).__init__()
        self.add_module('FirstLayer', FirstLayer(in_channel))
        self.add_module('Common_SeDensenet_parts', Common_SeDensenet_parts)

    def forward(self, x):
        out = self.FirstLayer(x)
        out = self.Common_SeDensenet_parts(out)

        return out


class SeDenseNet1(nn.Module):
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

        super(SeDenseNet1, self).__init__()

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
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

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

            if i != len(block_config) - 1:
                se_dict['features.transition{}.norm.weight'.format(i + 1)] = param_dict[
                    'features.transition{}.norm.weight'.format(i + 1)]

                se_dict['features.transition{}.norm.bias'.format(i + 1)] = param_dict[
                    'features.transition{}.norm.bias'.format(i + 1)]

                se_dict['features.transition{}.conv.weight'.format(i + 1)] = param_dict[
                    'features.transition{}.conv.weight'.format(i + 1)]

                nn.init.kaiming_normal_(se_dict['features.transition{}.conv_down.weight'.format(i + 1)])

                nn.init.kaiming_normal_(se_dict['features.transition{}.conv_up.weight'.format(i + 1)])

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
        out1 = features
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)
        out = self.classifier(out)
        weight = self.classifier.weight
        return out, out1, weight


class SeDenseNet2(nn.Module):
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

        super(SeDenseNet2, self).__init__()

        self.model_p = models.densenet121(pretrained=True)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
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
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

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
        nn.init.kaiming_normal_(se_dict['features.conv0.weight'])
        # se_dict['features.conv0.weight'] = param_dict['features.conv0.weight']
        se_dict['features.norm0.weight'] = param_dict['features.norm0.weight']
        se_dict['features.norm0.bias'] = param_dict['features.norm0.bias']
        for i, num_layer in enumerate(block_config):
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

            if i != len(block_config) - 1:
                se_dict['features.transition{}.norm.weight'.format(i + 1)] = param_dict[
                    'features.transition{}.norm.weight'.format(i + 1)]

                se_dict['features.transition{}.norm.bias'.format(i + 1)] = param_dict[
                    'features.transition{}.norm.bias'.format(i + 1)]

                se_dict['features.transition{}.conv.weight'.format(i + 1)] = param_dict[
                    'features.transition{}.conv.weight'.format(i + 1)]

                nn.init.kaiming_normal_(se_dict['features.transition{}.conv_down.weight'.format(i + 1)])

                nn.init.kaiming_normal_(se_dict['features.transition{}.conv_up.weight'.format(i + 1)])

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
        out1 = features
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)
        out = self.classifier(out)
        weight = self.classifier.weight
        return out, out1, weight


class End2EndSeDenseNet(nn.Module):
    def __init__(self):
        super(End2EndSeDenseNet, self).__init__()
        self.Firstnet = SeDenseNet1(growth_rate=32, block_config=(6, 12, 24, 16),
                                   num_init_features=64, bn_size=4, drop_rate=0, num_classes=6)
        self.Secondnet = SeDenseNet2(growth_rate=32, block_config=(6, 12, 24, 16),
                                    num_init_features=64, bn_size=4, drop_rate=0, num_classes=6)

    # def Img_synthesis(self, x, label, feature, weights):
    #     # _, feature, weights = self.Firstnet(x)
    #     h, w = x.shape[2:]
    #     out = torch.zeros([x.shape[0], 3, h, w]).cuda()
    #     attMap_out = torch.zeros([x.shape[0], 1, h, w]).cuda()
    #     for j in range(x.shape[0]):
    #         weight = weights[label[j], :]
    #         attMap = weight[0] * feature[j:j + 1, 0:1, :, :]
    #
    #         # out_new = attMap
    #         for i in range(1, len(weight)):
    #             attMap = attMap + weight[i] * feature[j:j + 1, i:i + 1, :, :]
    #         attMap_out[j, :, :, :] = F.interpolate(attMap, size=[h, w])
    #
    #     out = torch.cat((attMap_out, x), dim=1)
    #     return out

    def Img_synthesis(self, x, label, feature, weights):
        # _, feature, weights = self.Firstnet(x)
        h, w = x.shape[2:]
        out = torch.zeros([x.shape[0], 3, h, w]).cuda()
        attMap_out = torch.zeros([x.shape[0], 1, h, w]).cuda()
        for j in range(x.shape[0]):
            weight = weights[label[j], :]
            attMap = weight[0] * feature[j:j + 1, 0:1, :, :]

            # out_new = attMap
            for i in range(1, len(weight)):
                attMap = attMap + weight[i] * feature[j:j + 1, i:i + 1, :, :]
            attMap_out[j, :, :, :] = F.interpolate(attMap, size=[h, w])

        out = torch.cat((attMap_out, x), dim=1)
        return out

    def forward(self, x, flag):
        input = copy.deepcopy(x.detach()[:, 0:2, :, :])
        out1, feature, weights = self.Firstnet(x)
        pre = torch.max(torch.nn.functional.softmax(out1, dim=1), 1)[1]
        if flag is not False:
            label = flag
        else:
            label = pre
        feature_in = copy.deepcopy(feature.detach())
        weight_in = copy.deepcopy(weights.detach())
        input = self.Img_synthesis(input, label, feature_in, weight_in)
        out2 = self.Secondnet(input)
        return out1, out2[0]

    # def forward(self, x):
    #     out = self.FirstSeDensenet(x)
    #     ####
    #     out = self.SecondSeDensenet(out)
    #     return out

# if __name__ == '__main__':
#     model = AutoContextSeDenseNet()
#     optimzer1 = optim.Adam(model.FirstSeDensenet.parameters(), lr=0.01)
#     optimzer2 = optim.Adam(model.SecondSeDensenet.parameters(), lr=0.01)
