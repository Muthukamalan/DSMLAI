import torch
import math 



class ResidualConnection(torch.nn.Module):
    '''
        Residual connection
        x = x +  layer(x)
    '''
    def __init__(self, dropout_rate:float=0.01,verbose:bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dropout_rate:float = dropout_rate
        self.verbose:bool       = verbose

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.norm:LayerNormalization = LayerNormalization(verbose=self.verbose)

    def forward(self,x:torch.Tensor,sublayers:torch.nn.Module)->torch.Tensor:
        res:torch.Tensor = x + self.dropout(
                                        sublayers( 
                                            self.norm(x)
                                        )
                                )
        if self.verbose:
            print(f"Residual Connection tensor Changes from {x.shape} >> {res.shape}")
        return res

class InputEmbedding(torch.nn.Module):
    def __init__(self,vocab_size:int, d_model:int,verbose:bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size:int               = vocab_size
        self.d_model:int                  = d_model
        self.embedding:torch.nn.Embedding = torch.nn.Embedding(
                                                    num_embeddings=self.vocab_size,
                                                    embedding_dim=self.d_model
                                            )
        self.verbose:bool                 = False

    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
        args:
        - x:   
            [ [I,love,music], [I,love,art] ] >> [[1, 2, 4], [1, 2, 89]]   >> 
            
            [ 
                [   
                    [1,2,3,......512],
                    [1,2,3,......512],
                    [1,2,3,......512]   
                ] ,      
                [    
                    [1,2,3,......512],
                    [1,2,3,......512],
                    [1,2,3,......512]     
                ]  
            ]
        '''
        res = self.embedding(x) *  math.sqrt(self.d_model)
        if self.verbose:
            print(f"InputEmbedding tensor size changes:: {x.shape} >> {res.shape}")
        return res


class FeedForwardBlock(torch.nn.Module):
    '''
            # This is where COMPUTATIONAL part of Transformer comes in
            args::
                - d_model: inner dimension of a model and maintain same dim for next block
                - d_ff   : squeeze and expand kind of
                - dropout: float
    '''
    def __init__( self, d_model: int, d_ff: int, dropout: float=0.01,bias=False,verbose:bool=False, *args, **kwargs ) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=d_ff,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=d_ff, out_features=d_model,bias=bias),
        )
        self.verbose:bool = verbose

        if self.verbose:
            print(f"Internal dimensions of FeedForward Network changes from <{d_model}> >>  <{d_ff}> >> ReLU >> <{d_model}>")

    def forward(self, x: torch.Tensor)->torch.Tensor:
        '''
            working:: (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch,seq_len, d_model)
        '''
        res:torch.Tensor = self.linear(x)
        if self.verbose:
            print(f"Feedforward tensor size changes:: {x.shape} >> {res.shape}")
        return res


class LayerNormalization(torch.nn.Module):
    def __init__(self, eps:float=10e-6, verbose:bool=False, *args, **kwargs) -> None:
        super(LayerNormalization,self).__init__(*args, **kwargs)
        self.eps:float = eps
        self.verbose:bool = verbose
        self.alpha:torch.nn.Parameter = torch.nn.Parameter(torch.ones(1))
        self.bias:torch.nn.Parameter = torch.nn.Parameter(torch.zeros(1))


    def forward(self,x:torch.Tensor)->torch.Tensor:
        mean:torch.Tensor = x.mean(dim=-1,keepdim=True)
        std:torch.Tensor  = x.std(dim=-1,keepdim=True)
        res:torch.Tensor  = self.eps* (x - mean) / (std+self.eps) + self.bias
        if self.verbose:
            print(f"Layernorm tensor size changes:: {x.shape} >> {res.shape}")
        return  res
    

class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self,d_model:int, num_heads:int,droptout_rate:float=0.01, bias:bool=False, verbose:bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert d_model % num_heads ==0, "d_model should ne divisible by number of heads"

        self.d_model:int   = d_model
        self.num_heads:int = num_heads
        self.dropout_rate:int = droptout_rate
        self.verbose:bool  = verbose
        self.bias:bool     = bias


        self.d_k:int             = self.d_model // self.num_heads

        self.w_q:torch.nn.Linear = torch.nn.Linear(self.d_model,self.d_model,bias=self.bias)
        self.w_k:torch.nn.Linear = torch.nn.Linear(self.d_model,self.d_model,bias=self.bias)
        self.w_v:torch.nn.Linear = torch.nn.Linear(self.d_model,self.d_model,bias=self.bias)
        self.w_o:torch.nn.Linear = torch.nn.Linear(self.d_model,self.d_model,bias=self.bias)

    
    
    
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
        '''
            $$head_{i}= Attention(QW_{i}^{Q}  x KW_{i}^{K} x VW_{i}^{v})$$
            
            $$Multi-Head Attention (QxKxV) = Concat(head_{1},head_{2},..head_{n}) * W_{o}$$
        '''
        query:torch.Tensor = self.w_q(q)  # (batch, seq_len, )
        key:torch.Tensor   = self.w_k(k)
        value:torch.Tensor = self.w_v(v)

        # (batch_size, seq_len, d_model)   >> (batch_size,seq_len, heads, d_model//heads)  >> (batch_size, heads, seq_len, d_model//heads)
        query = query.view( query.shape[0], query.shape[1], self.num_heads, self.d_k ).transpose(1,2)
        key   = key.view  ( key.shape[0],   key.shape[1],   self.num_heads, self.d_k ).transpose(1, 2)
        value = value.view( value.shape[0], value.shape[1], self.num_heads, self.d_k ).transpose(1, 2)

        # Calculate Attention
        attn_output:torch.Tensor
        self.attention_scores:torch.Tensor
        attn_output ,self.attention_scores = MultiHeadAttentionBlock.attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)
        
        attn_output = attn_output.transpose(1,2).contiguous().view(attn_output.shape[0],-1,self.num_heads*self.d_k)
        # When you call contiguous(), it actually makes a copy of the tensor such that the order of its elements in memory is the same as if it had been created from scratch with the same data.
        # COMBINING ALL HEADS TOGETHER
        # (batch,head, seq_len, d_k) ---> (batch, seq_len, head , d_k ) ---> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) >> (batch, seq_len, d_model)
        res:torch.Tensor = self.w_o(attn_output)

        if self.verbose:
            print(f"Input For MultiHead Attention: k,q,v and mask {q.shape},{k.shape},{v.shape} and {mask.shape} respectively")
            print(f"After splits into Heads query:{query.shape}, key:{key.shape} and value:{value.shape}")
            print(f"passes into attention and it's shape:{attn_output.shape}")
            print(f"MultiHeadAttention tensor changes into >> {res.shape}")

        return res


    @staticmethod
    def attention(query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask:torch.Tensor,dropout_rate:float|None =0.01)->tuple[torch.Tensor,torch.Tensor]:
        '''
            (batch_size, token, seq_len, d_k ) --> (batch_size, token, seq_len, seq_len )
        '''
        d_k:int = query.shape[-1]   # Embedding DIM
        attention_score:torch.Tensor = ( query @ torch.transpose(key,-2,-1) )/ math.sqrt(d_k)     # Q * K.T  / √d_k

        if mask is not None:
            val = -1e9 if mask.dtype == torch.float32 else -1e+4
            attention_score.masked_fill(mask==0,val)   # Fill with float('-inf') 


        attention_score = attention_score.softmax(dim=-1)   #(batch_size, token, seq_len, seq_len )  apply softmax on embedding_dim
        
        if dropout_rate is not None:
            attention_score = torch.nn.functional.dropout(input=attention_score,p=dropout_rate)

        return (attention_score @ value),attention_score

class DecoderBlock(torch.nn.Module):
    def __init__(
            self,
            self_attention_block:MultiHeadAttentionBlock,
            cross_attention_block:MultiHeadAttentionBlock,
            feedforward_block:FeedForwardBlock ,
            dropout_rate:float =0.01,
            verbose:bool = False,
            *args, 
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attention_block:MultiHeadAttentionBlock                 = self_attention_block 
        self.cross_attention_block:MultiHeadAttentionBlock                = cross_attention_block
        self.feedforward_block:FeedForwardBlock                           = feedforward_block
        self.residual_connections:torch.nn.ModuleList[ResidualConnection] = torch.nn.ModuleList([ ResidualConnection(dropout_rate=dropout_rate) for _ in range(3)  ])
        self.verbose:bool                                                 = verbose 

    def forward(
            self, 
            x:torch.Tensor, 
            encoder_output:torch.Tensor,
            src_mask:torch.Tensor,
            target_mask:torch.Tensor
    )->torch.Tensor:
        '''
            x = x + self_attention_block(kqv=x,mask=src_mask)
            x = x + cross_attention_block(q=x,kv=encoder_output, mask=src_mask)
            x = x + feed_forward_block(x)
        '''
        x:torch.Tensor = self.residual_connections[0]( x, lambda x: self.self_attention_block( q=x, k=x,              v=x,              mask=target_mask) )
        x:torch.Tensor = self.residual_connections[1]( x, lambda x: self.cross_attention_block(q=x, k=encoder_output, v=encoder_output, mask=src_mask   ) )
        x:torch.Tensor = self.residual_connections[2]( x, self.feedforward_block )
        return x 
    

class Decoder(torch.nn.Module):
    '''
        Decoder Blocks
        - args:
            - layers: torch.nn.ModuleList
            - norm: LayerNormalization()

        - return: torch.FloatTensor

        - function
            - norm( decoder.forward(i) for i in decoder_layers )
    '''
    def __init__(self, layers:torch.nn.ModuleList ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder_layers:torch.nn.ModuleList[DecoderBlock] = layers
        self.norm:LayerNormalization                          = LayerNormalization()


    def forward(self,x:torch.Tensor, encoder_output:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor)->torch.Tensor:
        for dec in self.decoder_layers:
            x = dec(x,encoder_output, src_mask, tgt_mask)    # DecoderBlock.forward(x:PositionalEncoding, encoder_output, src_mask, tgt_mask)
        res:torch.Tensor = self.norm(x)
        return res
    

class EncoderBlock(torch.nn.Module):
    '''
    # Encoder Block
    - args:
        - attention_block:    MultiHeadAttentionBlock
        - feed_forward_block: FeedForwardBlock
        - dropout:            float
        - residual_connctions:torch.nn.ModuleList([ResidualConnection,ResidualConnection])
    - functionality
        - forward(x:torch.Tensor, src_mask:torch.Tensor)
    
    '''
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float=0.01,verbose:bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attention_block:MultiHeadAttentionBlock = self_attention_block
        self.feed_forward_block :FeedForwardBlock         = feed_forward_block 
        self.residual_connection:torch.nn.ModuleList      = torch.nn.ModuleList([ ResidualConnection() for _ in range(2) ])
        self.dropout_rate:float                           = dropout
        self.verbose:bool                                 = verbose

    def forward(self,x:torch.Tensor,src_mask:torch.Tensor)->torch.Tensor:
        '''
            x = x + self_attention_block(kqv=x,mask=src_mask)
            x = x + feed_forward_block(x)
        '''
        
        # res = x + self_attention_block(kqv=x,mask=src_mask)
        res:torch.Tensor = self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
         
        # res = res + forward_block(res)
        res:torch.Tensor = self.residual_connection[1](res,self.feed_forward_block)
        # 

        if self.verbose:
            print(f"EncoderBlock tensor change from {x.shape} >> {res.shape}, and src_mask::{src_mask.shape if src_mask is not None else ''}")
        return res
    

class Encoder(torch.nn.Module):
    '''
        Encoder Blocks
        - args:
            - layers: torch.nn.ModuleList[ EncoderBlock(s) ]
            - norm: LayerNormalization()

        - functions:
            - forward(
                    x: PositionalEmbedding,
                    mask: src_mask

            )
        - return: torch.FloatTensor


    '''
    def __init__(self,layers: torch.nn.ModuleList, verbose:bool=False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_layers:torch.nn.ModuleList[EncoderBlock] = layers                 # torch.nn.ModuleList( EncoderBlock(s)  )
        self.norm:LayerNormalization                          = LayerNormalization()
        self.verbose:bool                                     = verbose

    def forward(self,x:torch.Tensor,mask:torch.Tensor)->torch.FloatTensor:
        '''
        - args:
            - x: PositionalEmbedding
            - mask:
        '''
        for enc in self.encoder_layers:
            x = enc(x,mask)                       # EncoderBlock.forward(x=x,src_mask=mask )
        res:torch.Tensor = self.norm(x)

        if self.verbose:
            print(f"ENCODER tensor change from {x.shape} >> {res.shape}, and src_mask::{mask.shape if mask is not None else ''}")

        return res


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model:int, max_seq_len:int,  verbose:bool=False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.d_model:int        = d_model
        self.seq_len:int        = max_seq_len
        self.verbose:bool       = verbose

        pe:torch.Tensor = torch.zeros(self.seq_len,self.d_model)
        
        position:torch.Tensor = torch.arange(0,self.seq_len, dtype=torch.float).unsqueeze(1)  
        '''position::
            [
                1,
                2,
                3,
                .
                .
                max_seq_len
            ]
        '''

        div_term = torch.exp( torch.arange(0,self.d_model,2).float() * (-math.log(1e4)/self.d_model) )
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(dim=0)

        self.register_buffer(name='pe',tensor=pe)
        # pe should not to be considered a model parameter. 
        # For example, BatchNorm's running_mean is not a parameter, but is part of the module's state

    def forward(self,x:torch.Tensor)->torch.Tensor:
        res:torch.Tensor = x + (self.pe[:,:x.shape[1],:1]).requires_grad_(False)
        if self.verbose:
            print(f"PositionalEncoding tensor change from {x.shape} >> {res.shape}")
        return res



class ProjectionLayer(torch.nn.Module):
    def __init__(self, vocab_size:int, d_model:int, bias:bool=False, verbose:bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.d_model:int    = d_model
        self.vocab_size:int = vocab_size
        self.bias:bool      = bias 
        self.verbose:bool   = verbose

        self.projection:torch.nn.Linear = torch.nn.Linear(in_features=self.d_model,out_features=self.vocab_size,bias=self.bias,)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        res:torch.Tensor  = torch.log_softmax( self.projection(x), dim=-1 )
        if self.verbose:
            print(f"Projection Layer tensor size changes:: {x.shape} >> {res.shape}")
        return res
    

class Transformer(torch.nn.Module):
    def __init__(
            self, 
            encoder: Encoder,
            decoder:Decoder,
            src_embedded: InputEmbedding,
            tgt_embedded: InputEmbedding,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder:Encoder = encoder
        self.decoder:Decoder = decoder
        self.src_embedded: InputEmbedding  = src_embedded 
        self.tgt_embedded: InputEmbedding  = tgt_embedded
        self.src_pos: PositionalEncoding   = src_pos
        self.tgt_pos: PositionalEncoding   = tgt_pos 
        self.projection_layer: ProjectionLayer = projection_layer

    
    def encode(
            self,
            src:torch.Tensor,
            src_mask:torch.Tensor
    )->torch.Tensor:
        '''
            InputEmbedding >> PositionalEmbedding >> Encoder(x)
        '''
        src = self.src_embedded(src)        # src >> InputEmbedding<Tensor>::embedding(x) 
        src = self.src_pos(src)             # InputEmbedding<Tensor>::embedding(x)  >> PositionalEncoding(x)
        return self.encoder(src,src_mask)

    def decode(
            self,
            encoder_output:torch.Tensor,
            src_mask:torch.Tensor,
            tgt:torch.Tensor,
            tgt_mask:torch.Tensor
    )->torch.Tensor:
        '''
            InputEmbedding >> PositionalEmbedding >> Decoder(x)
        '''
        # (batch_size, seq_len , d_model )
        tgt = self.tgt_embedded(tgt)                               # src >> InputEmbedding<Tensor>::embedding(x) 
        tgt = self.tgt_pos(tgt)                                    # InputEmbedding<Tensor>::embedding(x)  >> PositionalEncoding(x)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
        

    def project( self, x:torch.Tensor  )->torch.Tensor:
        '''
            Projection(x)
        '''
        # ( batch_size, seq_len, vocab_size )
        res:torch.Tensor = self.projection_layer(x)
        return res