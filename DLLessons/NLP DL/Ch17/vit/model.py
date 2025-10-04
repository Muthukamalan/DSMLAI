import torch
import math 
import lightning as pl 
import torchvision




class PatchEmbedding(torch.nn.Module):
    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.
    """

    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super(PatchEmbedding, self).__init__()

        self.in_channels:int = in_channels
        self.patch_size:int = patch_size
        self.embedding_dim:int = embedding_dim
        self.patcher:torch.nn.Conv2d = torch.nn.Conv2d( in_channels=self.in_channels, out_channels=self.embedding_dim, kernel_size=self.patch_size, stride=patch_size,bias=False)
        self.flatten:torch.nn.Flatten = torch.nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Image Size must be divisible by patch size, given image shape: {image_resolution}, patch_size: {self.patch_size}"

        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)  # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]



class MultiheadSelfAttentionBlock(torch.nn.Module):
    """
    Creates a multi-head self-attention block
    """
    def __init__( self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0 )->None:
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
    def forward(self, x)->torch.Tensor:
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn( query=x, key=x, value=x, need_weights=False )
        return attn_output




class MLPBlock(torch.nn.Module):
    """
    Creates a Layer Normalized MLP Block
    """

    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            torch.nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    


class TransformerEncoderBlock(torch.nn.Module):
    """
    Creates a Transformer Encoder Block
    """

    def __init__(
        self,
        embedding_dim=768,
        num_heads=12,
        mlp_size=3072,
        mlp_dropout=0.1,
        attn_dropout=0,
    ):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x
    


class ViT(torch.nn.Module):
    def __init__(
            self, img_size:int = 224, 
            in_channels:int = 3, 
            patch_size:int = 16,
            num_transformer_layer:int = 12,
            embeddig_dim:int = 768,
            mlp_size:int = 3072,
            num_heads:int = 12,
            attn_dropout:int = 0,
            mlp_dropout:int = 0.01,
            embedding_dropout:int = 0.01, 
            num_classes:int = 1000,
            *args: torch.Any, 
            **kwargs: torch.Any
    ) -> None:
        super().__init__(*args, **kwargs)

        #( height*width  / patch^2 )
        assert (img_size%patch_size==0),f"Image Size must be divisible by patch size, given image shape: {img_size}, patch_size:{patch_size}"

        self.num_patches = (img_size*img_size) // patch_size**2

        self.class_embedding:torch.nn.Parameter = torch.nn.Parameter( 
                                                        data= torch.randn(1,1,embeddig_dim),
                                                        requires_grad= True
                                                )
        
        self.pos_embedding:torch.nn.Parameter = torch.nn.Parameter(
                                                        data = torch.randn(1, self.num_patches+1 , embeddig_dim),
                                                        requires_grad= True
                                                )
        
        self.patch_embedding:PatchEmbedding   = PatchEmbedding(in_channels=in_channels,patch_size=patch_size,embedding_dim=embeddig_dim)


        l = []
        for _ in range(num_transformer_layer):
            l.append(TransformerEncoderBlock(embedding_dim= embeddig_dim,num_heads=num_heads,mlp_size= mlp_size,mlp_dropout=mlp_dropout,attn_dropout=attn_dropout))
        



        
        # *[ TransformerEncoderBlock(embedding_dim= embeddig_dim,num_heads=num_heads,mlp_size= mlp_size, mlp_dropout=mlp_dropout,attn_dropout=attn_dropout ) for _ in range(num_transformer_layer) ]
        self.encoder:TransformerEncoderBlock =  torch.nn.Sequential(*[l[0],l[1],l[0],l[1],l[0],l[1],l[0],l[1]])  # torch.nn.Sequential(*l) 
        self.classifier:torch.nn.Sequential    = torch.nn.Sequential(
                                                        torch.nn.LayerNorm(normalized_shape=embeddig_dim),
                                                        torch.nn.Linear(in_features=embeddig_dim,out_features=num_classes)
                                                ) 
        
        self.sftmax:torch.nn.Softmax          = torch.nn.Softmax(dim=-1)

        self.embedding_dropout:torch.nn.Dropout = torch.nn.Dropout(p=embedding_dropout)


    def forward(self,x:torch.Tensor) -> torch.Any:
        '''
            x:: ( batch_size, C, H, W )
        '''
        batch_size = x.shape[0]
        class_token:torch.Tensor = self.class_embedding.expand(batch_size,-1,-1)  #(batch_size, -1, embedding_dim)
        x = self.patch_embedding(x)              # ( batch_size, N, P^2•C )
        x = torch.cat([class_token,x],dim=1)  
        x = self.pos_embedding + x
        x = self.embedding_dropout(x)
        x = self.encoder(x)
        x = self.classifier(x[:,0])
        x = self.sftmax(x)
        return x
    


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: torch.nn.init.zeros_(m.bias)