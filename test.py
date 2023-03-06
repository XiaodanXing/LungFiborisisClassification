import pandas as pd
import torch
import os

from utils.AveMeter import ParseAndChecArguments
from torchvision import transforms
from dataset import biDatasetGeneral
import numpy as np
from PIL import Image
import torch.nn as nn
import importlib
import csv
from sklearn.metrics import classification_report



import timm


net_names = {
    '0':['fibrosisv2','/media/NAS03/xiaodan/pathology/imgs',2,1,1024,
                  ['uip','prob','indeter','other']],


             }

for k,net_name in net_names.items():

    filename,  filepath, class_num, channel_num, resolution, class_names = net_name
    print(filename)
    model_ft = timm.create_model('inception_resnet_v2', pretrained=True, drop_rate=0.5)
    model_ft.classif = nn.Linear(1536, class_num)
    model_ft.conv2d_1a = nn.Conv2d(channel_num, 32, (3, 3), (2, 2), bias=False)


    ckpt_name = './ckpts/%s/checkpoints/ckpt_000010.pth' % filename

    test_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])

    ])



    model_ft = nn.parallel.DataParallel(model_ft)
    model_ft = model_ft.cuda()

    densenet121 = torch.load(ckpt_name)
    pretrained_dict = densenet121['model_state_dict']
    model_ft.load_state_dict(pretrained_dict)

    test_dataset = biDatasetGeneral(filepath=filepath,purpose='test', transforms=test_transform,

                                          class_names=class_names,num_channels=channel_num,)

    test_loaders = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=1)
    # loader = test_loaders

    model_ft.eval()
    pred_list = []
    label_list = []
    confid_list = []
    for batch_idx, (input, target, input_name), in enumerate(test_loaders):
        # Create vaiables
        target = target.view(-1, target.shape[-1])
        # target = torch.LongTensor(np.squeeze(target))
        target = torch.LongTensor(target.view(-1))

        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            # compute output
            output = model_ft(input_var)
            confid = torch.nn.functional.softmax(output, dim=1)



            _, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
            correct = torch.sum(pred == target_var)
            pred_list.append(pred.cpu().numpy())
            label_list.append(target_var.cpu().numpy())
            confid_list.append(confid.cpu().numpy())
            a = 1

    pred_arr = np.concatenate(pred_list, axis=0)
    label_arr = np.concatenate(label_list, axis=0)
    confid_arr = np.concatenate(confid_list, axis=0)
    prediction_arr = np.asarray(pred_arr)
    label_arr = np.asarray(label_arr)



    print(classification_report(label_arr,prediction_arr))
    result = np.concatenate([prediction_arr.reshape(prediction_arr.shape[0],1),
                             label_arr.reshape(prediction_arr.shape[0],1)],axis=1)
    result = np.concatenate([result,confid_arr],axis=1)
    columns = ['pred','true',] + class_names
    result = pd.DataFrame(result,columns=columns)

    result.to_csv('./results/%s.csv'%(str(filename)))
