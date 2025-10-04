import os 
import math 
import torch

from typing import Dict

def google_bert_gelu(x:torch.Tensor)->torch.Tensor:
    return 0.5 * x *  ( 1.+
                        torch.tanh( 
                                math.sqrt(2./math.pi) * (x+.044715*torch.pow(x,3.))
                            )
                        )

class CasualSelfAttention(torch.nn.Module):
    def __init__(self,config:Dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 

        for k,v in config.items(): self.__setattr__(k,v) 

        assert self.n_embed % self.n_head ==0
        self.attn_layer:torch.nn.Linear = torch.nn.Linear(self.n_embed, 3*self.n_embed)  # x >>> k,q,v
        self.out:torch.nn.Linear = torch.nn.Linear(self.n_embed, self.n_embed)    # k,q,v > out 

        self.attn_dropout:torch.nn.Dropout = torch.nn.Dropout(self.dropout_rate)
        self.resid_dropout:torch.nn.Dropout= torch.nn.Dropout(self.dropout_rate)

        # casual mask to ensure that attention is only applied to the left in the input seq
        self.register_buffer('bias',tensor= torch.tril(torch.ones(self.block_size,self.block_size)).view(1, 1, self.block_size, self.block_size) )
        '''
            block_size=10
                [[[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]]

            # Batch-1, Seq-1, Mask-(10,10) 
        '''



    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
            input (bs,seq_len,embedding_dim)  >> output (bs,seq_len,embedding_dim)

            x     :: (bs,seq_len,embedding_dim)
            attn  :: (bs, seq_len, 3*embedding_dim)
            .split:: (bs, seq_len, 3*embedding_dim).split(embedding_dim,dim=2)    
            # Each chunk (bs,seq_len,embedding) is a view of the original tensor, split across embeddin_dim so, 3 will get

            k,q,v >> (bs,seql_len, n_heads, embedding_dim//n_heads) >> (bs,head, seql_len, embedding_dim//n_heads)
            # Each Heads are responsible for different context of seq_len

        '''
        B,T,C = x.size()        # Batch, Seq_len, Embedding_dimension
        
        # calc q,k,v
        q:torch.Tensor;
        k:torch.Tensor;
        v:torch.Tensor;
        q,k,v  = self.attn_layer(x).split(split_size=self.n_embed,dim=2)  # each with n_embed  i.e) q (32,1024,768)
        
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2) 
        k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2) 
        v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2) 

        attn = (q @ k.transpose(-2,-1)) * (1./math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        attn = torch.nn.functional.softmax(attn,dim=-1)
        attn = self.attn_dropout(attn)
        y:torch.Tensor    = attn @ v   # (bs, n_heads, T,T) @ (bs, n_heads, T, embding_dm/n_heads ) >> (bs,n_heads, seq_len, embedding_dim/n_heads )
        y:torch.Tensor    = y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_dropout(self.out(y))



class MLP(torch.nn.Module):
    def __init__(self,config:Dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        for k,v in config.items(): self.__setattr__(k,v)

        self.fc:torch.nn.Linear   = torch.nn.Linear(in_features= self.n_embed, out_features= 4*self.n_embed)
        self.proj:torch.nn.Linear = torch.nn.Linear(in_features= 4*self.n_embed, out_features= self.n_embed)
        self.dropout:torch.nn.Dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
            input(bs,seq_len,embedding_dim)  >> output(bs,seq_len,embedding_dim)
        '''
        x = google_bert_gelu(self.fc(x))   # relu(fc(x))
        return self.dropout(self.proj(x))



class Block(torch.nn.Module):
    def __init__(self,config:Dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config 
        for k,v in config.items(): self.__setattr__(k,v)
        self.ln_1:torch.nn.LayerNorm = torch.nn.LayerNorm
        self.attn:CasualSelfAttention= CasualSelfAttention(config)
        self.ln_2:torch.nn.LayerNorm = torch.nn.LayerNorm
        self.mlp:MLP                 = MLP(config)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = x+self.attn(self.ln_1(x) )
        x = x+self.mlp(self.ln_2(x))
        return x
        


class NanoGPT(torch.nn.Module):
    def __init__(self,config:Dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        for k,v in config.items(): 
            '''
                - vocab_size, n_embed, n_block_size, dropout_rate, n_layer
            '''
            self.__setattr__(k,v) 

        self.transformer = torch.nn.ModuleDict(
            dict(
                txt_embed = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.n_embed ),
                pos_embed = torch.nn.Embedding(num_embeddings=self.n_block_size, embedding_dim=self.n_embed ),
                drop      = torch.nn.Dropout(p=self.dropout_rate),
                h = torch.nn.ModuleList([
                    Block(self.config) for _ in range(self.n_layer)
                ]),
                ln = torch.nn.LayerNorm(self.n_embed)
            )
        )
        self.lm_head = torch.nn.Linear(self.n_embed, self.vocab_size,bias=False )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        b,t = x.size()
        device = x.device
        assert t<=self.block_size, f"cannot forward seq of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0,t,dtype=torch.long,device=device).unsqueeze(0)   #(1,t)

        # GPT Model
        token_emb = self.transformer.txt_embed(x)
        posit_emb = self.transformer.pos_embed(pos)
        x = self.transformer.drop( token_emb+posit_emb )
        for _block in self.transformer.h: 
            x = _block(x)
        x = self.transformer.ln(x)
        # logits
        return self.lm_head(x)   