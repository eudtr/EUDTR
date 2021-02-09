import torch
import numpy
from utils import data_distance_random
import tslearn.metrics as metrics


def student_t_euc(samples, pos_samples, neg_samples, cuda):
        
    
    distance = torch.zeros(neg_samples.size(0) + 1, neg_samples.size(1))
    distance = distance.cuda() if cuda else distance
    
    #calculating similarity
    #positive
    distance[0, :] = torch.sqrt(torch.sum(torch.square(samples.view(samples.size(0), -1) - pos_samples.view(pos_samples.size(0), -1)), axis=1))

    #neigtive
    for i in range(neg_samples.size(0)):
        distance[i + 1, :] = torch.sqrt(torch.sum(torch.square(samples.view(samples.size(0), -1) - neg_samples[i].view(neg_samples[i].size(0), -1)), axis=1))

    alpha = 1
    # calculate the distance between x and x_
    q = 1.0 / (1.0 + distance / alpha)
    q = q.pow((alpha + 1.0) / 2.0) 
    q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, axis=1), 0, 1)
    
    return q




class MseLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cuda):

        super(MseLoss, self).__init__()
        # self.nb_random_samples = nb_random_samples
        self.cuda = cuda

    def forward(self, batch, neg_samples, encoder, decoder, save_memory=False):
        
        feature = encoder(batch)
        pos_samples = decoder(feature)

        q = student_t_euc(batch, pos_samples, neg_samples, self.cuda)
        # loss of positive sample
        loss1 = - torch.mean(torch.log(q[0, :]))
        # loss of Negative sample
        loss2 = - torch.mean(torch.log(1 - q[1:, :]))
        loss = loss1 + loss2 
        
        if save_memory:
            loss.backward()
            del q
            torch.cuda.empty_cache()

        

        
        
        