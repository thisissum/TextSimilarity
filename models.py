import torch 
from torch import nn
from modules.layers import LocalInferenceLayer


class ESIM(nn.Module):
    """Implementation of 'Enhanced LSTM for Natural Language Inference'
    Use BiGRU and local inference mechanisim to compute similarity of 2 sentence
    """
    def __init__(self, model_args):
        super(ESIM, self).__init__()
        model_args.inject_to(self)
        self.BiGRU_Encoder = nn.GRU(
            self.emb_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True, 
            dropout=self.dropout, 
            bidirectional=True)
        self.LocalInferenceLayer = LocalInferenceLayer()
        self.BiGRU_Out = nn.GRU(
            self.hidden_dim*2*4, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True, 
            dropout=self.dropout, 
            bidirectional=True)
        self.MaxPooling = nn.AdaptiveMaxPool1d(1)#will attach to lastest layer
        self.AvgPooling = nn.AdaptiveAvgPool1d(1)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.hidden_dim*2*4, self.output_dim)
    
    def forward(self, emb_seq1, emb_seq2):
        #encoding
        encoded_seq1, _ = self.BiGRU_Encoder(emb_seq1)
        encoded_seq2, _ = self.BiGRU_Encoder(emb_seq2)
        #local inference
        seq1_hat, seq2_hat = self.LocalInferenceLayer(encoded_seq1, encoded_seq2)
        #compute inter feature and concate
        cat_a = torch.cat(
            [encoded_seq1, seq1_hat, encoded_seq1 - seq1_hat, encoded_seq1 * seq1_hat], dim=-1)
        cat_b = torch.cat(
            [encoded_seq2, seq2_hat, encoded_seq2 - seq2_hat, encoded_seq2 * seq2_hat], dim=-1)
        #feed into another bigru
        encoded_cat_a, _ = self.BiGRU_Out(cat_a)
        encoded_cat_b, _ = self.BiGRU_Out(cat_b)
        #pooling though all seq_len
        max_pool_cat_a = self.MaxPooling(encoded_cat_a.permute(0,2,1)).squeeze(-1)
        max_pool_cat_b = self.MaxPooling(encoded_cat_b.permute(0,2,1)).squeeze(-1)
        avg_pool_cat_a = self.AvgPooling(encoded_cat_a.permute(0,2,1)).squeeze(-1)
        avg_pool_cat_b = self.AvgPooling(encoded_cat_b.permute(0,2,1)).squeeze(-1)
        #final output
        final_vec = torch.cat(
            [max_pool_cat_a,max_pool_cat_b,avg_pool_cat_a,avg_pool_cat_b], dim=-1)
        output = self.fc(self.tanh(final_vec))
        return output


