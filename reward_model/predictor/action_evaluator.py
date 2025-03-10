import torch
import torch.nn as nn
from transformers import AutoModel


class ActionEvaluator(nn.Module):
    def __init__(self, emb_dim=384, n_heads=8, hidden_dim=512, n_layers=6, dropout=0.1):
        super(ActionEvaluator, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=n_heads, num_encoder_layers=n_layers, 
                                       num_decoder_layers=n_layers, dim_feedforward=hidden_dim, dropout=dropout)
        # self.fc_out = nn.Linear(emb_dim, output_dim)

    def forward(self, src_embedded, trg_embedded):
        output = self.transformer(src_embedded, trg_embedded)
        # output = self.fc_out(transformer_out)
        return output