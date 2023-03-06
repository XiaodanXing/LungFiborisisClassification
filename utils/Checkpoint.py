import os
import torch
import glob

def loadLatestCheckpoint(ckpt_dir, net, optimizer):
    '''

    :param ckpt_dir:
    :param net:
    :param optimizer:
    :return:
    '''
    checkpoint_file_list = glob.glob(os.path.join(ckpt_dir, 'ckpt_*.pth'))
    if not checkpoint_file_list:
        print("No checkpoint files are found.Start training from scratch")
        return 0

    checkpoint_file_list.sort()
    labtest_checkpoint_file = checkpoint_file_list[-1]

    # load checkpoint
    ckpt = torch.load(labtest_checkpoint_file)
    net.load_state_dict(ckpt['model_state_dict'])

    return ckpt['epoch_index'] + 1


def SaveCheckpoint(checkpoint_dir, net, optimizer, epoch_index, is_best=False):
    '''

    :param checkpoint_dir:
    :param net:
    :param optimizer:
    :param epoch_index:
    :param is_best:
    :return:
    '''
    ckpt = {'epoch_index': epoch_index,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}

    if is_best:
        latest_checkpoint_file = os.path.join(checkpoint_dir,
                                              'best_ckpt.pth')
    else:
        latest_checkpoint_file = os.path.join(checkpoint_dir,
                                              "ckpt_{0:06d}.pth".format(epoch_index))
    torch.save(ckpt, latest_checkpoint_file)
