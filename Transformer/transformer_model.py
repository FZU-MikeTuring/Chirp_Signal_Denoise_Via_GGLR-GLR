import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import math

class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)

class PositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                              (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0) # (1, seq_len, embed_dim) 多加一个批次维度，方便后续计算

"""
A=softmax((Q @ K^T)/sqrt(d_k))

output=A @ V
"""
def attention(Q,K,V,mask=None,dropout=None):
    d_k=Q.size(-1)

    #torch.matmul() 只计算最低两维的矩阵乘法，其他维度保持不变
    scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k) # scores的维度是(batch_size, num_heads, q_seq_len, k_seq_len)
    if mask is not None:
        scores=scores.masked_fill(mask==False,float('-inf'))
    
    attention_weights=torch.softmax(scores,dim=-1) # A的维度是(batch_size, num_heads, q_seq_len, k_seq_len)
    if dropout is not None:
        attention_weights=dropout(attention_weights)

    output=torch.matmul(attention_weights,V)

    return output,attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout:float=None):
        super(MultiHeadAttention,self).__init__()

        assert embed_dim % num_heads == 0,"embed_dim must be divisible by num_heads"

        self.d_k=embed_dim//num_heads
        self.num_heads=num_heads
        self.d_model=embed_dim

        self.Wq=nn.Linear(embed_dim,embed_dim)
        self.Wk=nn.Linear(embed_dim,embed_dim)
        self.Wv=nn.Linear(embed_dim,embed_dim)
        self.Wo=nn.Linear(embed_dim,embed_dim)

        if dropout is not None:
            self.dropout=nn.Dropout(dropout)
        else :
            self.dropout=None

    def forward(self,Q,K,V,mask=None):
        batch_size=Q.size(0)

        if mask is not None:
            if mask.dim()==2: # (q_seq_len, k_seq_len)
                mask=mask.unsqueeze(0).unsqueeze(0) # (1, 1, q_seq_len, k_seq_len)
            elif mask.dim()==3:
                mask=mask.unsqueeze(1) # (batch_size, 1, q_seq_len, k_seq_len)
            
            if mask.size(1)==1:
                mask=mask.expand(batch_size,self.num_heads,-1,-1) # (batch_size, num_heads, q_seq_len, k_seq_len)

        # 因为直接线性变换后先按行分开，所以变成四维度是(batch_size, seq_len, num_heads, d_k)，然后转置成(batch_size, num_heads, seq_len, d_k)方便后续计算
        Q=self.Wq(Q).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K=self.Wk(K).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V=self.Wv(V).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2) 

        # 计算多头注意力 
        # attn_output的维度是(batch_size, num_heads, q_seq_len, d_k)，attn_weights的维度是(batch_size, num_heads, q_seq_len, k_seq_len)
        attn_output, attn_weights = attention(Q, K, V, mask=mask, dropout=self.dropout)

        # 将多头输出拼接回原来的维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.Wo(attn_output)

        return output, attn_weights

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim,dropout:float=None):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout= nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x, pad_mask=None):
        attn_output, attn_weights = self.attention(x, x, x, mask=pad_mask)
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)

class Encoder(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_heads, ff_dim, dropout: float = None):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)])

    def forward(self, x, pad_mask=None):
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout: float = None):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout= nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x, enc_output, cross_mask=None, look_ahead_mask=None):
        self_attn_output, _ = self.self_attention(x, x, x, mask=look_ahead_mask)
        if self.dropout is not None:
            self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)
        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, mask=cross_mask)
        if self.dropout is not None:
            cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm2(x + cross_attn_output)
        ffn_output = self.ffn(x)
        return self.norm3(x + ffn_output)
    
class Decoder(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_heads, ff_dim, dropout: float = None):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)])

    def forward(self, x, enc_output, cross_mask=None, look_ahead_mask=None):
        for block in self.blocks:
            x = block(x, enc_output, cross_mask=cross_mask, look_ahead_mask=look_ahead_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_encoder_blocks, num_decoder_blocks=0, dropout: float = None, encoder_only: bool = False):
        super(Transformer, self).__init__()
        self.encoder_only = encoder_only
        self.embedding = Embedding(input_dim, embed_dim)
        self.position_encoding = PositionEncoding(embed_dim)
        self.encoder = Encoder(num_encoder_blocks, embed_dim, num_heads, ff_dim, dropout)
        if not self.encoder_only:
            self.decoder = Decoder(num_decoder_blocks, embed_dim, num_heads, ff_dim, dropout)
        else:
            self.decoder = None
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, src, tgt=None, src_pad_mask=None, tgt_cross_mask=None, look_ahead_mask=None):
        if self.encoder_only:
            src_embedded = self.position_encoding(self.embedding(src))
            enc_output = self.encoder(src_embedded, pad_mask=src_pad_mask)
            return self.output_layer(enc_output)

        src_embedded = self.position_encoding(self.embedding(src))
        enc_output = self.encoder(src_embedded, pad_mask=src_pad_mask)
        tgt_embedded = self.position_encoding(self.embedding(tgt))
        dec_output = self.decoder(tgt_embedded, enc_output, cross_mask=tgt_cross_mask, look_ahead_mask=look_ahead_mask)
        return self.output_layer(dec_output)
    
