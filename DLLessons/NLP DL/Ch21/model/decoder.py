import torch
from .feedforward import FeedForwardBlock
from .layernorm import LayerNormalization



class MultiHeadAttention(torch.nn.Module):
    def __init__(
            self, 
            d_model:int,
            seq_len,
            head_size,
            n_head,
            bias,
            attn_dropout
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList(
                            [Attention(
                                    d_model=d_model,
                                    seq_len=seq_len,
                                    head_size=head_size,
                                    bias=bias,
                                    attn_dropout=attn_dropout
                                ) for _ in range(n_head)]
        )
        self.out  = torch.nn.Linear(d_model,d_model,bias=False)
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self,x):
        mha = torch.cat([head(x) for head in self.heads],dim=-1)
        out = self.dropout(self.out(mha))
        return out

class Attention(torch.nn.Module):
    def __init__(
            self, 
            d_model,
            seq_len, 
            head_size:int,
            bias:bool=False,
            attn_dropout=0.01,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.c_attn = torch.nn.Linear(d_model, 3 * head_size, bias=bias)
        self.register_buffer('tril',torch.tril(torch.ones(seq_len,seq_len)))
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        B,T,C = x.shape  # (batch_size, seq_len, d_model)
        q, k, v  = self.c_attn(x).split(self.head_size, dim=-1)
        # q (bs,seq_len, d_model//head)
        # k (bs,seq_len, d_model//head)
        # v (bs,seq_len, d_model//head)
        attn_score = q @ k.transpose(-2,-1) * C**-.5                    # (bs,seq_len, seq_len)
        attn_score = attn_score.masked_fill(self.tril[:T,:T]==0,1e-9)   # tril:: (seq_len,seq_len)
        attn_score = torch.nn.functional.softmax(attn_score,dim=-1)
        attn_score = self.dropout(attn_score)
        return attn_score@v


class DecoderBlock(torch.nn.Module):
    def __init__(
            self,
            d_model:int,
            n_head:int,
            seq_len:int,
            bias:bool=False,
            attn_dropout:float=0.01,
    ) -> None:
        super().__init__()

        assert d_model%n_head==0,'d_model dim must be divisible by number of heads'

        head_size:int  = d_model // n_head
        self.multi_attn = MultiHeadAttention(
                                d_model=d_model,
                                seq_len=seq_len,
                                head_size=head_size,
                                n_head=n_head,
                                bias=bias,
                                attn_dropout=attn_dropout
        )
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.feedforward = FeedForwardBlock(d_model=d_model,d_ff=4*d_model,dropout=attn_dropout,bias=bias)
    
    def forward(self,x):
        x = x + self.multi_attn(self.layernorm1(x))
        x = x + self.feedforward(self.layernorm2(x))
        return x