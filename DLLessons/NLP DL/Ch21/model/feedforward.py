import torch 

class FeedForwardBlock(torch.nn.Module):
    '''
            # This is where COMPUTATIONAL part of Transformer comes in
            args::
                - d_model: inner dimension of a model and maintain same dim for next block
                - d_ff   : squeeze and expand kind of
                - dropout: float
    '''
    def __init__( self, d_model: int, d_ff: int, dropout: float,bias=False, *args, **kwargs ) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=d_ff,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=d_ff, out_features=d_model,bias=bias),
        )

    def forward(self, x: torch.Tensor):
        '''
            working:: (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch,seq_len, d_model)
        '''
        return self.linear(x)