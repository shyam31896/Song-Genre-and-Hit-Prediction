import os
import re
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--construct",type=str2bool, nargs='?', help="Extract data from sources and construct the dataset")
parser.add_argument("-t", "--train",type=str2bool, nargs='?', help="Train the models to find the best performing one and test it on the Test data")
args = parser.parse_args()
if args.construct:
	dataset_creation = True
else:
	dataset_creation = False
if args.train:
	do_train = True
else:
	do_train = False

if dataset_creation == True:
	print('Dataset Construction started')
	os.system('python3 dataset_constructor.py')
	print('Datasets successfully constructed')

if do_train == True:
	print('Training started')
	os.system('python3 train.py')

os.system('python3 validate.py')

os.system('python3 test.py')
'''

Datasets are stored in data/
Models are stored in the models/ directory
Intermediate training results are stored in conf_matrices/

'''
	
