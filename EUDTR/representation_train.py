import math
import numpy
import torch
import sklearn.metrics as sm
import sklearn.externals
import sklearn.model_selection
from sklearn.cluster import KMeans
import metrics
import utils
import triplet_loss
import reconstruction_loss
import network
from utils import data_distance_random
import time


class CausalCNNClusterer:
    """
    Training process of time series representation

    compared_length: Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    nb_random_samples: Number of randomly chosen intervals to select the
           final negative sample in the loss.
    negative_penalty: Multiplicative coefficient for the negative sample
           loss.
    batch_size: Batch size used during the training of the network.
    epochs: Number of optimization steps to perform for the training of
           the network.
    lr: learning rate of the Adam optimizer used to train the network.

    channels: Number of channels manipulated in the causal CNN.
    depth: Depth of the causal CNN.
    reduced_size: Fixed length to which the output time series of the
           causal CNN is reduced.
    out_channels: Number of features in the final output.
    kernel_size: Kernel size of the applied non-residual convolutions.
    in_channels: Number of input channels of the time series.
    cuda: Transfers, if True, all computations to the GPU.
    gpu: GPU index to use, if CUDA is enabled.
    """

    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, epochs=30, lr=0.001,channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0, n_clusters=1, dim=1, n_init=10, timestep=1):


        self.in_channels = in_channels
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_clusters=n_clusters
        self.encoder = network.CausalCNNEncoder(in_channels, channels, depth, reduced_size, out_channels,kernel_size).double()
        self.decoder = network.Discriminator(out_channels, reduced_size, dim, timestep, kernel_size, stride=1, padding=1, out_padding=0).double()
        self.cuda = cuda
        self.encoder = self.encoder.cuda() if self.cuda else self.encoder
        self.decoder = self.decoder.cuda() if self.cuda else self.decoder
        self.triplet_loss = triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty, self.cuda
        )
        self.mse_loss = reconstruction_loss.MseLoss(self.cuda)
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.dim=dim
        self.nb_random_samples = nb_random_samples

    def train(self, X, y, save_memory=False):

        #the preprocessing of train set
        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda()
        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )


        for epoch in range(self.epochs):
            epoch_start=time.time()
            for batch_num,batch in enumerate(train_generator):
                if self.cuda:
                    batch = batch.cuda()
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                #Get negative samples
                neg_samples = torch.from_numpy(data_distance_random(train, batch, self.nb_random_samples))
                neg_samples = neg_samples.cuda() if self.cuda else neg_samples

                #Joint optimization of network parameters with triple loss and MSE loss
                self.triplet_loss(batch, self.encoder, train, neg_samples, save_memory=save_memory)
                self.mse_loss(batch, neg_samples, self.encoder, self.decoder, save_memory)
        
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

            epoch_end=time.time()
            print('train--Epoch: ', epoch + 1, " time: ", epoch_end - epoch_start)

            features = self.encode(X, self.batch_size)
            km = KMeans(n_clusters=self.n_clusters, n_init=self.n_init).fit(features.reshape(features.shape[0], -1))
            y_pred = km.labels_

            result = metrics.cluster_evaluation(y, y_pred)
            print("train--Epoch: {}---result: {}".format(epoch + 1, result))


        return self
    
    def encode(self, X, batch_size=50):

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size)
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda()
                features[
                count * batch_size: (count + 1) * batch_size
                ] = self.encoder(batch).cpu()
                count += 1

        self.encoder = self.encoder.train()
        return features

    def set_params(self, compared_length, nb_random_samples, negative_penalty, batch_size,
                   epochs, lr, channels, depth,reduced_size, out_channels,
                   kernel_size, in_channels, cuda, gpu, n_clusters, dim, n_init, timestep):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            epochs, lr, channels, depth,reduced_size, out_channels,
            kernel_size, in_channels, cuda, gpu, n_clusters, dim, n_init, timestep
        )
        return self