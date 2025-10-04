import torch
import math


class PositionalEncoding(torch.nn.Module):
    '''
        PositionalEncoding = InputEncoding + PositionalValues
        args:
        - d_model :
        - seq_len :
        - dropout :
    '''
    def __init__( self, d_model: int, seq_len: int, dropout: float, *args, **kwargs ) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = torch.nn.Dropout(p=dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  #(seq_len, 1)
        
        # vecor of shape (d_mdoel)
        div_term = torch.exp( torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model) )

        # Sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) #  sin(position * (10000* (2i/d_model   ) ))

        # cosine to odd indices 
        pe[:, 1::2] = torch.cos(position * div_term)  # cos( position * (10000* ( 2i/d_model ) ))

        # add batch dimension to the positional encoding    
        pe = pe.unsqueeze(0)   # (1, seq_length, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)   #( batch_size, seq_len:`x.shape[1]`, d_model    )
        return self.dropout(x)


