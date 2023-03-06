import argparse
import os

import torch
from torchvision import transforms

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument("--name", type=str,default='fibrosis',
                            help="name of this task")
        self.parser.add_argument("--save_dir", type=str,
                                 default='./ckpts/',

                            help="save directory")
        self.parser.add_argument("--resume_epoch", type=bool, default=False,
                            help="resume training")
        self.parser.add_argument("--gpu_ids", default='2',
                            help="choose gpu device.")
        # dataset parameters
        self.parser.add_argument("--filepath", type=str,
                                 default='./CTMontage',
                            help="input filepath")
        self.parser.add_argument("--train_folder", type=str,
                                 default='train',
                            help="input filepath")
        self.parser.add_argument("--class_name", default='NORMAL,PNEUMONIA',
                            help="names of classes")
        self.parser.add_argument("--image_size", type=int,default=512,
                            help="size of image")
        self.parser.add_argument("--num_threads", default=4,
                                 help="number of loader threads")
        self.parser.add_argument("--num_channels", type=int,default=1,
                                 help="input channel")

        ## loss
        self.parser.add_argument("--loss_name", default='CrossEntropyLoss',
                            help="input filepath")
        ### net
        self.parser.add_argument("--model", default='inception_resnet_v2',
                                 help="densenet121,efficientnet_b3a,inception_resnet_v2")

        ##train
        # training parameters
        self.parser.add_argument("--vis_on", default=False,type=bool,
                                 help="visdom visualization")
        self.parser.add_argument("--epochs", type=int,default=100,
                                 help="num of epochs")
        self.parser.add_argument("--batch_size", type=int,default=2,
                                 help="batch size")
        self.parser.add_argument("--lr", default=0.0001,
                                 help="learning rate")
        self.parser.add_argument("--betas", default=(0.9, 0.999),
                                 help="betas")
        self.parser.add_argument("--plot_snapshot", default=50,
                                 help="snapshot")
        self.parser.add_argument("--save_epochs", default=100,
                                 help="snapshot")

        self.parser.add_argument("--optim", default='Adam',
                                 help="snapshot")

        self.parser.add_argument("--augment", default='none',
                                 help="augmentation type:none, transform,erase,mixup")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.str_gpu_ids = self.opt.gpu_ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)


        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])



        class_ids = self.opt.class_name.split(',')
        self.opt.class_name = []
        for class_id in class_ids:
            self.opt.class_name.append(class_id)

        if self.opt.augment == 'none':
            self.opt.train_transform = transforms.Compose([
                transforms.Resize((self.opt.image_size,self.opt.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        elif self.opt.augment == 'transform':
            self.opt.train_transform = transforms.Compose([
                # transforms.RandomAffine(15, scale=(0.9, 1.1)),
                transforms.Resize((self.opt.image_size,self.opt.image_size)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.7, saturation=0.7),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),

            ])





        print('------------ Options -------------')
        for k, v in sorted(self.opt.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.output_dir = os.path.join(self.opt.save_dir, self.opt.name)

        return self.opt
