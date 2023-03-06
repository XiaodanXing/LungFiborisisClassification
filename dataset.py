import copy
import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

from PIL import Image,ImageOps
import torch.nn as nn
from sklearn import decomposition
import nibabel as nib



class biDatasetGeneral(Dataset):

    def __init__(self,filepath ,
                 class_names = ['uip','prob','indeter','other'],
                 transforms = None,purpose='train',num_channels=1):  # crop_size,


        self.imlist = []

        self.filepath = os.path.join(filepath,purpose)


        for path, subdirs, files in os.walk(self.filepath):
            max_idx = len(files) * self.r
            idx = 0
            for name in files:
                if '.json' not in name and idx <= max_idx:
                    self.imlist.append(os.path.join(path, name))
                    idx += 1


        self.class_names = class_names
        self.transforms = transforms
        self.num_class = len(class_names)
        self.num_channels = num_channels

    def __getitem__(self, idx):
        if idx < len(self.imlist):
            img_name = self.imlist[idx]
            filename = os.path.split(img_name)[-1]

            img = Image.open(img_name)
            if self.num_channels ==1:
                img = ImageOps.grayscale(img)
            if self.transforms is not None:
                img = self.transforms(img)

            label = self.class_names.index(img_name.split('/')[-2])


            return img,label,filename



    def __len__(self):
        return len(self.imlist)

