import torch
import os
from utils.sec_helper import LoadConfig
from utils.AveMeter import ParseAndChecArguments
from torchvision import transforms
from dataset import biDataset
import numpy as np
from PIL import Image
import torch.nn as nn
import importlib
import csv
from sklearn.metrics import classification_report

args = ParseAndChecArguments()
config_file = args.config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = LoadConfig(config_file)

test_transform = transforms.Compose([
    transforms.Resize((cfg.dataset.resize,cfg.dataset.resize)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = biDataset(purpose='test',transforms=test_transform

)

test_loaders = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=cfg.train.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=cfg.train.num_threads)
# loader = test_loaders
batch_size = cfg.train.batch_size



import timm

for net_name in ['inception_resnet_v2','densenet121','efficientnet_b3a']:
    for pretrain in [42,1024,1]:
        if net_name == 'inception_resnet_v2':
            model_ft = timm.create_model('inception_resnet_v2', pretrained=bool(pretrain==1), drop_rate=0.5)
            model_ft.classif = nn.Linear(1536, 4)


        if net_name == 'densenet121':
            model_ft = timm.create_model('densenet121', pretrained=bool(pretrain==1), drop_rate=0.5)
            model_ft.classifier = nn.Linear(1024, 4)

        if net_name == 'efficientnet_b3a':
            model_ft = timm.create_model('efficientnet_b3a', pretrained=bool(pretrain==1), drop_rate=0.5)
            model_ft.classifier = nn.Linear(1536, 4)

        gpu_ids = range(cfg.general.num_gpus)
        model_ft = nn.parallel.DataParallel(model_ft, device_ids=gpu_ids)
        model_ft = model_ft.cuda()

        pred_list = []
        label_list = []
        for batch_idx, (input, target, input_name), in enumerate(test_loaders):
            # Create vaiables
            target = target.view(-1, target.shape[-1])
            # target = torch.LongTensor(np.squeeze(target))
            target = torch.LongTensor(target.view(-1))

            with torch.no_grad():
                input_var = torch.autograd.Variable(input.cuda())
                target_var = torch.autograd.Variable(target.cuda())

                # compute output
                output = model_ft(input_var.repeat(1,3,1,1))

                _, pred = output.max(1)
                correct = torch.sum(pred == target_var)
                pred_list.append(pred.cpu().numpy())
                label_list.append(target_var.cpu().numpy())
                a = 1

        pred_arr = np.concatenate(pred_list, axis=0)
        label_arr = np.concatenate(label_list, axis=0)
        prediction_arr = np.asarray(pred_arr)
        label_arr = np.asarray(label_arr)

        print(net_name, pretrain)
        print(classification_report(label_arr,prediction_arr))
