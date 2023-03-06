from utils.AveMeter import AverageMeter
import time
import torch
import numpy as np
from utils.sec_helper import calculate_acc



def train_epoch(model, loader, optimizer, epoch, n_epochs, loss_func, logger, num_classes, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()


    # train_loss = 0
    correct = 0
    total = 0
    # Model on train model
    model.train()
    pred_list = []
    label_list = []
    end = time.time()
    for batch_idx, (input, target, _) in enumerate(loader):

        target = target.view(-1, target.shape[-1])
        # target = torch.LongTensor(np.squeeze(target,axis=1))

        target = torch.LongTensor(target.view(-1))
        # -----------
        # for i in range(input.shape[0]):
        #     label = target[i].cpu().numpy()
        #     feature = get_map(input[i, :, :, :], label)
        #     feature = np.asarray(feature)
        #     feature = torch.from_numpy(feature)
        #     input[i, 0, :, :] = feature

        input_var = torch.autograd.Variable(input.cuda())
        input_var.requires_grad = True
        target_var = torch.autograd.Variable(target.cuda())
        # inlabel = target.cpu().numpy()
        # # inlabel = False
        output = model(input_var)
        # output1, output2 = model(input_var, inlabel)
        # if epoch < 100:
        #     output = output1
        # else:
        # output = output2
        loss = loss_func(output, target_var)

        # measure accuracy and record loss
        batch_size = target.size(0)
        # print(output)
        _, pred = output.max(1)
        correct = torch.sum(pred == target_var)
        pred_list.append(pred.cpu().numpy())
        label_list.append(target_var.cpu().numpy())

        # print(correct.item())
        error.update(torch.ne(pred.cpu().squeeze(), target).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        # if epoch <100:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    pred_arr = np.concatenate(pred_list, axis=0)
    label_arr = np.concatenate(label_list, axis=0)
    prediction_arr = np.asarray(pred_arr)
    label_arr = np.asarray(label_arr)

    # epoch_auc, auc_each_class = calculate_auc(prediction_arr, label_arr, num_classes)
    epoch_acc, acc_each_class = calculate_acc(prediction_arr, label_arr, num_classes)
    for acc in acc_each_class:
        logger.info('{}'.format(acc))

    msg = 'epoch: {}, floss: {:.4f}, error: {:.4f}, time: {:.4f} s/vol'
    msg = msg.format(epoch, losses.avg, error.avg, batch_time.avg)
    logger.info(msg)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, epoch_acc, acc_each_class


def test_epoch(model, loader, loss_func, logger, epoch, num_classes, ratio=4, print_freq=100, is_test=True, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    target_label = []
    pred_label = []
    # test_loss = 0
    # total = 0
    # correct = 0
    model.eval()
    pred_list = []
    label_list = []

    end = time.time()

    for batch_idx, (input, target, input_name), in enumerate(loader):

        # Create vaiables
        target = target.view(-1, target.shape[-1])
        # target = torch.LongTensor(np.squeeze(target))
        target = torch.LongTensor(target.view(-1))
        # for i in range(input.shape[0]):
        #     label = target[i].cpu().numpy()
        #     feature = get_map(input[i, :, :, :], label)
        #     feature = np.asarray(feature)
        #     feature = torch.from_numpy(feature)
        #     input[i, 0, :, :] = feature

        inlabel = False

        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            # compute output
            # output1, output2 = model(input_var, inlabel)
            # if epoch < 100:
            #     output = output1
            # else:
            #     output = output2
            output = model(input_var)
            # loss = torch.nn.functional.cross_entropy(output, target_var)
            loss = loss_func(output, target_var)

        if test:
            pre = torch.max(torch.nn.functional.softmax(output, dim=1), 1)[1].data.cpu().numpy().tolist()

            if batch_idx == 0:
                input_names = input_name
                targets = target.numpy().tolist()
                pres = pre
            else:
                input_names += input_name
                targets += target.numpy().tolist()
                pres += pre

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        target_label.extend(target.cpu().numpy().tolist())
        pred_label.extend(pred.cpu().numpy().squeeze(axis=1).tolist())
        pred_list.append(pred.cpu().numpy())
        label_list.append(target_var.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    pred_arr = np.concatenate(pred_list, axis=0)
    label_arr = np.concatenate(label_list, axis=0)
    prediction_arr = np.asarray(pred_arr)
    label_arr = np.asarray(label_arr)

    # epoch_auc, auc_each_class = calculate_auc(prediction_arr, label_arr, num_classes)
    epoch_acc, acc_each_class = calculate_acc(prediction_arr, label_arr, num_classes)
    for acc in acc_each_class:
        logger.info('{}'.format(acc))
    msg = 'epoch: {}, floss_var: {:.4f}, error_var: {:.4f}, time: {:.4f} s/vol'
    msg = msg.format(epoch + 1, losses.avg, error.avg, batch_time.avg)
    logger.info(msg)
    if test:
        return batch_time.avg, losses.avg, error.avg, input_names, targets, pres,
    else:
        # Return summary statistics
        return batch_time.avg, losses.avg, error.avg, epoch_acc, target_label, pred_label, acc_each_class
