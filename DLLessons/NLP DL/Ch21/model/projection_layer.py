import torch


class ProjectionLayer(torch.nn.Module):
    '''
        # Projection Layer
        x = nn.Linear(d_model,vocab_size)
        torch.log_softmax(x,dim=-1)
    '''
    def __init__(self, d_model:int, vocab_size:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.projection = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)
