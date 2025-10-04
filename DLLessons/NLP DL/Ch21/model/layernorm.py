import torch

class LayerNormalization(torch.nn.Module):
    '''
        alpha * {(x-mean)/sqrt(var+eps)}  +  beta
    '''
    def __init__(self, eps=10**-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor)->torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta