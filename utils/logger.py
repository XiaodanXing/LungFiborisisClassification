import logging
import os

def SetupLogger(log_file):
    '''

    :param log_file:
    :return:
    '''
    logging_dir = os.path.dirname(log_file)
    if not os.path.isdir(logging_dir):
        os.makedirs(logging_dir)

    logger = logging.getLogger('ad_densnet')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    h1 = logging.StreamHandler()
    h1.setLevel(logging.DEBUG)
    h1.setFormatter(formatter)
    logger.addHandler(h1)

    # create file handle
    h2 = logging.FileHandler(log_file)
    h2.setLevel(logging.DEBUG)
    h2.setFormatter(formatter)
    logger.addHandler(h2)

    return logger
