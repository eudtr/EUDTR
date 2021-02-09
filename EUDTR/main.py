import os
from time import time
import json
import math
import torch
import numpy
import argparse
from sklearn.cluster import KMeans
import representation_train
import metrics
import warnings
import random

warnings.filterwarnings('ignore')

#Set random seed
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def fit_hyperparameters(file, train, train_labels, cuda, gpu, dim, timestep, data_name,
                        save_memory=False):

    clusterer = representation_train.CausalCNNClusterer()

    clusterer.set_data_name(data_name)
    #Load parameters from the parameters file of the data
    hf = open(file, 'r')
    params = json.load(hf)
    hf.close()

    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    params['dim'] = dim
    params['timestep'] = timestep
    clusterer.set_params(**params)
    print(params)
    return clusterer.train(train, train_labels, save_memory),params


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--data', type=str, default='ArticularyWordRecognition',
                        help='the name of data set')
    parser.add_argument('--cuda', action='store_true',default=True,
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=1, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, default='_hyperparameters.json', metavar='FILE',
                        help='path of the file of hyperparameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--seed', type=int, default=13, help='a int number')
    parser.add_argument('--isTranspose', type=int, default=0, help='the shape of input data is (batch, dim, timestep)')
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arguments()
    seed_torch(args.seed)
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")

    data_name = args.data
    # load data
    path = '../data/{}/'.format(data_name)
    X_train_path = path + 'x_train.npy'
    y_train_path = path + 'y_train.npy'
    X_train = numpy.load(X_train_path)
    y_train = numpy.load(y_train_path)
    print("X_train.shape:", X_train.shape)

    if args.isTranspose:
        X_train = X_train.transpose(0, 2, 1)

    print("X_train.shape:", X_train.shape)
    train = X_train
    y_train = y_train.astype(numpy.float64)
    train_labels = numpy.squeeze(y_train)

    print(numpy.unique(train_labels))

    dim=train.shape[1]
    timestep = train.shape[-1]
    hyper ="../hyper/" + args.data + "/" + args.data+args.hyper

    t0 = time()
    clusterer,params = fit_hyperparameters(
        hyper, train, train_labels, args.cuda, args.gpu, dim, timestep, data_name,
        save_memory=True
    )

    print('Training time: ', (time() - t0))
    y_pred = clusterer.init_cluster_weights(train)
    result = metrics.cluster_evaluation(train_labels, y_pred)
    print("Clustering result：",result)