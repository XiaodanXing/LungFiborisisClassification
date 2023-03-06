import os
import shutil
from utils.Checkpoint import loadLatestCheckpoint, SaveCheckpoint
from utils.logger import SetupLogger
from utils.AveMeter import ParseAndChecArguments
import numpy as np
import torch
from torchvision import transforms
from torch.backends import cudnn
import matplotlib
from config.options import BaseOptions
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import biDatasetGeneral
from sklearn.metrics import classification_report, confusion_matrix
from visdom import Visdom
from train import train_epoch,test_epoch


def main():
    opt = BaseOptions().parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.str_gpu_ids
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    np.random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)

    if not opt.resume_epoch:
        shutil.rmtree(opt.output_dir)

    # enable logging
    log_file = os.path.join(opt.output_dir, 'logging', 'train_log.txt')
    logger = SetupLogger(log_file)

    # enable cudnn
    cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise EnvironmentError('CUDA is not availabel!Please check nvidia driver')



    num_classes = len(opt.class_name)


    train_dataset = biDatasetGeneral(filepath=opt.filepath,
        purpose=opt.train_folder,transforms=opt.train_transform,
        class_names= opt.class_name,num_channels=opt.num_channels,
    )
    val_dataset = biDatasetGeneral(filepath=opt.filepath,
        purpose='val',transforms=opt.train_transform,
        class_names= opt.class_name,num_channels=opt.num_channels,

    )


    train_loaders = DataLoader(train_dataset,
                               batch_size=opt.batch_size,
                               shuffle=True,
                               pin_memory=False,
                               num_workers=opt.num_threads)
    test_loaders = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=opt.batch_size,
                                               # shuffle=True,
                                               pin_memory=False,
                                               num_workers=opt.num_threads)


    import timm
    if opt.model == 'inception_resnet_v2':
        model_ft = timm.create_model('inception_resnet_v2', pretrained=False,drop_rate=0.5)
        model_ft.classif = nn.Linear(1536,num_classes)
        model_ft.conv2d_1a = nn.Conv2d(opt.num_channels,32,(3,3),(2,2),bias=False)

    if opt.model == 'densenet121':
        model_ft = timm.create_model('densenet121', pretrained=False,drop_rate=0.5)
        model_ft.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model_ft.classifier = nn.Linear(1024,num_classes)

    if opt.model== 'efficientnet_b3a':
        model_ft = timm.create_model('efficientnet_b3a', pretrained=False, drop_rate=0.5)
        model_ft.classifier = nn.Linear(1536, num_classes)
        model_ft.conv_stem = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    gpu_ids = opt.gpu_ids
    model_ft = nn.parallel.DataParallel(model_ft, device_ids=gpu_ids)
    model_ft = model_ft.cuda()

    # create optimizer

    if opt.optim == 'Adam':
        optimizer = optim.Adam(model_ft.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=1e-3)  # 0.4


        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
        # scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', factor=0.5, patience=4, verbose=True)

    elif opt.optim == 'SGD':
        optimizer = optim.SGD(model_ft.parameters(), lr=opt.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # Create loss function
    if opt.loss_name == 'CrossEntropyLoss':
        # loss_func = nn.CrossEntropyLoss()
        # weight = torch.cuda.FloatTensor([1, 2, 4, 4, 2])
        # loss_func = nn.CrossEntropyLoss(weight=weight)

        loss_func = nn.CrossEntropyLoss()
        # weight = torch.cuda.FloatTensor([1, 4, 1, 1, 2, 1])
        # loss_func = nn.CrossEntropyLoss()
        # loss_func = nn.CrossEntropyLoss()
    else:
        raise ValueError('Unsupported loss function')


    checkpoint_dir = os.path.join(opt.output_dir, 'checkpoints')
    if opt.resume_epoch is not True:
        if os.path.isdir(checkpoint_dir):
            raise ValueError('Directory exist, '
                         'o you need to re-train %s?' % os.path.split(opt.output_dir)[-1])
        else:
            os.makedirs(checkpoint_dir)
            epoch_start = 0
    else:
        epoch_start = loadLatestCheckpoint(checkpoint_dir, model_ft, optimizer)




    ratio = 1
    best_error = 1
    best_pred_label = []


    # --------------------------HE-------------------------#

    if opt.vis_on:
        vis = Visdom() #use_incoming_socket=False
        assert vis.check_connection()
        win_loss = vis.line(np.arange(10))  # create the window
        win_acc = vis.line(np.arange(10))
        win_train_acc = vis.line(np.arange(10))
        win_test_acc = vis.line(np.arange(10))
        x_index = []
        loss = [[], []]
        acc = [[], []]
        train_accs = [[] for i in range(num_classes)]
        validate_accs = [[] for i in range(num_classes)]

    # --------------------------HE-------------------------#

    for epoch_start in range(epoch_start, epoch_start + opt.epochs):
        # if epoch_start < 100:
        #     optimizer = optimizer1
        #     scheduler = scheduler1
        # else:
        #     optimizer = optimizer2
        #     scheduler = scheduler2
        _, train_loss, train_error, train_acc, train_acc_each = train_epoch(
            model=model_ft,
            loader=train_loaders,
            optimizer=optimizer,
            epoch=epoch_start,
            n_epochs=opt.epochs,
            loss_func=loss_func,
            num_classes=num_classes,
            logger=logger

        )
        print(opt.output_dir, optimizer.state_dict()['param_groups'][0]['lr'])

        _, test_loss, test_error, test_acc, target_label, pred_label, test_acc_each = test_epoch(
            model=model_ft,
            loader=test_loaders,
            loss_func=loss_func,
            epoch=epoch_start,
            num_classes=num_classes,
            logger=logger,
            ratio=ratio,
            is_test=(not test_loaders)
        )
        class_report = classification_report(target_label, pred_label)
        print(class_report)
        if optimizer.state_dict()['param_groups'][0]['lr'] > 1e-15:
            scheduler.step(test_loss)


        if epoch_start % 10 == 0 and epoch_start > 0:
            SaveCheckpoint(checkpoint_dir, model_ft, optimizer, epoch_start, is_best=False)
        if test_loaders and test_error < best_error:
            best_error = test_error



            print('New best error:%.4f' % best_error)
            SaveCheckpoint(checkpoint_dir, model_ft, optimizer, epoch_start, is_best=True)


        # --------------------------HE-------------------------#
        if opt.vis_on:
            x_index.append(epoch_start)
            loss[0].append(train_loss)
            loss[1].append(test_loss)
            acc[0].append(train_acc)
            acc[1].append(test_acc)
            for i in range(num_classes):
                train_accs[i].append(train_acc_each[i])
                validate_accs[i].append(test_acc_each[i])

            vis.line(X=np.column_stack((np.array(x_index) for i in range(2))),
                     Y=np.column_stack((np.array(loss[i]) for i in range(2))),
                     win=win_loss,
                     opts=dict(title='LOSS',
                               xlabel='epoch',
                               xtick=1,
                               ylabel='loss',
                               markersymbol='dot',
                               markersize=5,
                               legend=['train loss', 'test loss']))
            vis.line(X=np.column_stack((np.array(x_index) for i in range(2))),
                     Y=np.column_stack((np.array(acc[i]) for i in range(2))),
                     win=win_acc,
                     opts=dict(title='ACCURACY',
                               xlabel='epoch',
                               xtick=1,
                               ylabel='accuracy',
                               markersymbol='dot',
                               markersize=5,
                               legend=['train acc', 'test acc']))

            vis.line(X=np.column_stack((np.array(x_index) for i in range(num_classes))),  # 6
                     Y=np.column_stack((np.array(train_accs[i]) for i in range(num_classes))), # 6

                     win=win_train_acc,
                     opts=dict(title='TRAIN ACCURACY',
                               xlabel='epoch',
                               xtick=1,
                               ylabel='accuracy',
                               markersymbol='dot',
                               markersize=5,
                               legend=opt.class_name)  #
                               # legend=['GD', 'jiaozeng', 'linba', 'zhuanyi', 'yanzheng', 'Glioma'])
                     )

            vis.line(X=np.column_stack((np.array(x_index) for i in range(num_classes))),
                     Y=np.column_stack((np.array(validate_accs[i]) for i in range(num_classes))),
                     win=win_test_acc,
                     opts=dict(title='TEST ACCURACY',
                               xlabel='epoch',
                               xtick=1,
                               ylabel='accuracy',
                               markersymbol='dot',
                               markersize=5,
                               legend=opt.class_name)  #
                               # legend=['GD', 'jiaozeng', 'linba', 'zhuanyi', 'yanzheng', 'Glioma'])
                     )





if __name__ == '__main__':


    main()
