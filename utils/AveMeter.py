import argparse
import os


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ParseAndChecArguments():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(os.getcwd(), 'config', 'config.py')

    print("default config = {0}".format(default_config))

    parser.add_argument('--config', type=str, default=default_config,
                        help="Folder containing the detection training config file")

    args = parser.parse_args()
    invalid_argument = False

    if not args.config:
        print('Please specify the configuration file')
        invalid_argument = True
    elif not os.path.isfile(args.config):
        print("The specified config:{0} does not exist!".format(args.config))
        invalid_argument = True

    if invalid_argument:
        raise ValueError("Invalid input arguments!")

    return args
