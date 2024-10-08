import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=500):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Stander_Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self,  d_src, d_hidden,n_layers, n_head,   dropout=0.1, n_position=500, scale_emb=False):

        super().__init__()
        d_k=d_v=d_model=d_inner=d_hidden

        
        self.position_enc = PositionalEncoding(d_hidden, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.embedding = nn.Sequential(nn.Linear(d_src, d_hidden),nn.Dropout(p=dropout),nn.ReLU())

    def forward(self, src_seq,return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        src_seq=self.embedding(src_seq)

        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=None)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
        
class MHA(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self,  d_src, d_hidden,n_layers, n_head,   dropout=0.1, n_position=500, scale_emb=False):

        super().__init__()
        d_k=d_v=d_model=d_inner=d_hidden

        
        self.position_enc = PositionalEncoding(d_hidden, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
       
        self.encoder = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.embedding = nn.Sequential(nn.Linear(d_src, d_hidden),nn.Dropout(p=dropout),nn.ReLU())

    def forward(self, src_seq,return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        src_seq=self.embedding(src_seq)

        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)
        enc_output, enc_slf_attn = self.encoder(enc_output, enc_output, enc_output,)


        return enc_output
        
class ATT(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self,  hidden_size,dropout):

        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear( hidden_size,  1),
            nn.Dropout(p=dropout),
            nn.Sigmoid(),
        )

    def forward(self, feature):

        att=self.attention(feature).permute(0,2,1)
        att=F.softmax(att,dim=1)
        feature=torch.bmm(att,feature).squeeze()


        return feature
            

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        
        n_z=64
        
        self.z_x = nn.Linear(2*d_model, n_head * n_z)
        self.z_b = nn.Linear(2*d_model, n_head * n_z, bias=False)

        d_x = [nn.Linear(n_z, d_model, bias=False) for _ in range(n_head)]
        self.d_x = nn.ModuleList(d_x)
        d_b = [nn.Linear(n_z, d_model) for _ in range(n_head)]
        self.d_b = nn.ModuleList(d_b)

        self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(d_model, d_v)) for _ in range(n_head)])
        for i in range(n_head):
            nn.init.orthogonal_(self.w_x[i])
            


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
        
        
        
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn