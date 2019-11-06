import torch
from torch import nn
import torch.nn.functional as F

class LocalInferenceLayer(nn.Module):
    """Local inference modeling part of ESIM
    """

    def __init__(self):
        super(LocalInferenceLayer, self).__init__()

    def forward(self, seq_1, seq_2):
        #seq_1.shape = batch_size, seq_1_len, feature_dim
        #seq_2.shape = batch_size, seq_2_len, feature_dim

        # batch_size, seq_1_len, seq_2_len
        e_ij = torch.matmul(seq_1, seq_2.permute(0, 2, 1))

        # weighted for inference
        weighted_seq2 = F.softmax(e_ij, dim=2)
        weighted_seq1 = F.softmax(e_ij, dim=1)

        # inference
        seq_1_hat = torch.matmul(weighted_seq2, seq_2)  # same shape as seq_1
        seq_2_hat = torch.matmul(weighted_seq1.permute(0, 2, 1), seq_1)

        return seq_1_hat, seq_2_hat