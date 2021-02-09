import numpy
import torch.utils.data
import tslearn.metrics as metrics

class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    # Arguments
    dataset: Numpy array representing the dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


class LabelledDataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset and its associated labels.

    # Arguments
    dataset: Numpy array representing the dataset.
    labels: One-dimensional array of the same length as dataset with
           non-negative int values.
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]


def data_distance_random(X_train, batch, nb_random_samples):
    """

   # Arguments
    X_train: all train data and the type of it is numpy array
    batch: a batch train data
    nb_random_samples:the number of negative samples for each sample in batch

    #return
    negative samples
    """

    ratio = 0.2
    X_train = X_train.cpu().numpy()

    # Randomly select a certain proportion of samples as negative samples
    # No repeat sampling
    index = numpy.random.choice(X_train.shape[0], int(X_train.shape[0] * ratio), False)
    X_train = X_train[index]
    
    batch = batch.cpu().numpy()
    dist = numpy.zeros((batch.shape[0], X_train.shape[0]))
    dist_sort = numpy.zeros((batch.shape[0], nb_random_samples))
    neg = numpy.zeros((nb_random_samples, batch.shape[0], batch.shape[1], batch.shape[2]))
    for i in range(batch.shape[0]):
        for j in range(X_train.shape[0]):
            dist[i, j] = metrics.dtw(X_train[i], X_train[j])
    for i in range(batch.shape[0]):
        index = numpy.argsort(-dist[i])
        dist_sort[i] = index[:nb_random_samples]
    dist_sort = dist_sort.astype('int64')
    for j in range(nb_random_samples):
        neg[j] = X_train[dist_sort[:, j]]
    return neg
