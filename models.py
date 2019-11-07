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


class InferSent(nn.Module):
    """Implementation of InferSent paper from facebook
    Use BiLSTM as encoder and compute inter connection by abs and abstract
    """
    def __init__(self, model_args):
        super(InferSent, self).__init__()
        self.model_args = model_args
        model_args.inject_to(self)
        self.BiLSTM_Encoder = nn.LSTM(
            self.emb_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=self.dropout, 
            batch_first=True, 
            bidirectional=True,)
        self.MaxPooling = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(
            2 * self.hidden_dim * 4, 512), 
            nn.Tanh(),
            nn.Linear(512, self.output_dim))
    
    def forward(self, emb_seq1, emb_seq2):
        encoded_seq1, _ = self.BiLSTM_Encoder(emb_seq1)
        encoded_seq2, _ = self.BiLSTM_Encoder(emb_seq2)
        infer_out = torch.cat([
            encoded_seq1, 
            encoded_seq2, 
            torch.abs(encoded_seq1 - encoded_seq2), 
            encoded_seq1 * encoded_seq2],
            dim=2)
        pooled_infer = self.MaxPooling(infer_out.permute(0,2,1)).squeeze(-1)
        output = self.fc(pooled_infer)
        return output

class SiameseNet(nn.Module):
    """
    """
    def __init__(self, model_args):
        super(SiameseNet, self).__init__()
        model_args.inject_to(self)
        self.LSTM_Encoder = nn.LSTM(
            self.emb_dim, 
            self.hidden_dim, 
            self.num_layers,
            dropout=self.dropout, 
            batch_first=True, 
            bidirectional=True)
        self.dense = nn.Linear(self.hiddem_dim*2, 128)
        self.AvgPooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, emb_seq1, emb_seq2):
        encoded_seq1, _ = self.LSTM_Encoder(emb_seq1)
        encoded_seq2, _ = self.LSTM_Encoder(emb_seq2)
        avg1 = self.AvgPooling(encoded_seq2.permute(0,2,1)).squeeze(-1)
        avg2 = self.AvgPooling(encoded_seq2.permute(0,2,1)).squeeze(-1)
        after_compress1 = self.dense(avg1)# batch_size, 128
        after_compress2 = self.dense(avg2)# batch_size, 128
        dot_product = (after_compress1 * after_compress2).sum(dim=-1)
        length_product = after_compress1.pow(2).sum(-1).pow(0.5) * after_compress2.pow(2).sum(-1).pow(0.5)
        output = dot_product / length_product
        return output