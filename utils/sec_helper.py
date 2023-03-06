import os
import importlib
import numpy as np


def LoadConfig(config_file):
    config_path = os.path.dirname(config_file)
    config_base = os.path.basename(config_file)
    config_name, _ = os.path.splitext(config_base)
    os.sys.path.insert(0, config_path)
    lib = importlib.import_module(config_name)
    os.sys.path.pop(0)

    return lib.cfg


def LoadModel(model_file):
    model_path = os.path.dirname(model_file)
    model_base = os.path.basename(model_file)
    model_name, _ = os.path.splitext(model_base)
    os.sys.path.insert(0, model_file)
    model = importlib.import_module(model_name)
    os.sys.path.pop(0)

    return model


def calculate_acc(y_pred, y_gt, num_classes):
    '''calculate the mean AUC'''
    acc_num_each_class = np.zeros((num_classes))
    num_each_class = np.zeros((num_classes))

    for k, label in enumerate(y_gt):
        num_each_class[label] += 1
        if y_pred[k] == label:
            acc_num_each_class[label] += 1

    return acc_num_each_class.sum() / num_each_class.sum(), acc_num_each_class / num_each_class
