import torch
import numpy
from utils import data_distance_random

class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series.

    compared_length: Maximum length of randomly chosen time series. If None, this parameter is ignored.
    nb_random_samples: Number of negative samples per batch example.
    negative_penalty: Multiplicative coefficient for the negative sample loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty, cuda):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.cuda = cuda

    def forward(self, batch, encoder, train, neg_samples, save_memory=False):
        batch_size = batch.size(0)
        length = min(self.compared_length, train.size(2))

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(1, high=length + 1)

        #length of anchors
        random_length = numpy.random.randint(
            length_pos_neg, high=length + 1
        )

        # Start of anchors
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )

        # start of positive samples in the anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )

        # start of another positive samples in the anchors
        beginning_samples_pos_local = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        ) 

        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos

        # Start of another positive samples in the batch examples
        beginning_positive_local = beginning_batches + beginning_samples_pos_local

        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        # End of another positive samples in the batch examples
        end_positive_local = beginning_positive_local + length_pos_neg


        # Start of negative samples
        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )

        # Anchors representations
        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ))

        # Positive samples representations
        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))

        # another positive samples representations
        positive_local_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive_local[j] - length_pos_neg: end_positive_local[j]
            ] for j in range(batch_size)]
        ))


        size_representation = representation.size(1)

        # Positive loss: -logsigmoid of dot product between anchor and positive representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # local positive loss: -logsigmoid of dot product between positive representations
        # and another positive representations in same anchors
        loss += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            positive_representation.view(batch_size, 1, size_representation),
            positive_local_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss
        # and free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            del positive_local_representation
            torch.cuda.empty_cache()


        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between anchor and negative representations
            neg_sample = neg_samples[i]
            negative_representation = encoder(
                torch.cat([neg_sample[
                    j:j+1, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            if save_memory:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss

