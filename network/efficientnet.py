from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import importlib



# def get_eff_model(num_classes):
#     # load an instance segmentation model pre-trained pre-trained on COCO
#     model = EfficientNet.from_pretrained('efficientnet-b0')
#     model = nn.parallel.DataParallel(model)
#
#     # get number of input features for the classifier
#
#     model._fc
#
#     return model


def FE():
    model = EfficientNet.from_pretrained('efficientnet-b5')
    model = nn.parallel.DataParallel(model)
    model.cuda()
    # model._avg_pooling = nn.AdaptiveAvgPool2d(1)
    # model._dropout = nn.Dropout(0.4, inplace=False)
    model.module._fc = nn.Linear(2048, 6, bias=True)
    return model

class EffNet(nn.Module):#attention sedensenet
    def __init__(self):
        super(EffNet, self).__init__()
        # self.L = 500
        # self.D = 128
        # self.K = 1


        self.feature_extractor_part1 = FE()



        self.instance_classifier = nn.Sequential(
            nn.Softmax()
        )





    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        # print(H.shape)



        IL = self.instance_classifier(H)



        return IL