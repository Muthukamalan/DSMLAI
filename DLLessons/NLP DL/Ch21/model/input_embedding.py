import torch
import math


class InputEmbeddings(torch.nn.Module):
    '''
        args:
        - d_model   : inner dimension of a model and maintain same dim for next block
        - vocab_size: total number of words in dictionary

    '''
    def __init__(self, d_model: int, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.tensor):
        '''
            (batch, seq_length) ---> (batch, seq_length, d_model)
            ```
            [                            [
              write a poem,                    [ [0.2194, -0.3953, -0.1042,  0.4274], [0.5886, -0.3110,  0.1537,  0.6280],  [0.2294, -0.3933, -0.842,  0.65]  ],
              sing a song,                     [ [0.4339,  0.3474,  0.5908, -0.2519], [0.0300, -0.0066,  0.1505,  0.0680],  [0.294, -0.3353, -0.92,  0.53] ],
              lit the lamp                     [ [ 0.0201,  0.6114,  0.9871,  0.416], [ 0.1894,  0.2805,  0.4651,  0.515],  [0.4, -0.39, -0.42,  0.344]      ],
            ]                           ] 
            ```
        '''
        return self.embedding(x) * math.sqrt(self.d_model)